import torch
from torch import nn as nn
from timm.layers import to_2tuple
import scipy.stats as stats

from finworld.registry import ENCODER
from finworld.registry import EMBED
from finworld.models.encoder.base import Encoder
from finworld.models import GRUBlock as Block
from finworld.models import DiagonalGaussianDistribution

@ENCODER.register_module(force=True)
class VAELSTMEncoder(Encoder):
    def __init__(self,
                 *args,
                 embed_config: dict = None,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 256,
                 depth: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 norm_layer = nn.LayerNorm,
                 cls_embed: bool = True,
                 sep_pos_embed: bool = True,
                 trunc_init: bool = False,
                 no_qkv_bias: bool = False,
                 if_mask: bool = False,
                 if_remove_cls_embed = True,
                 mask_ratio_min: float = 0.5,
                 mask_ratio_max: float = 1.0,
                 mask_ratio_mu: float = 0.55,
                 mask_ratio_std: float = 0.25,
                 **kwargs
                 ):
        super(VAELSTMEncoder, self).__init__()

        self.data_size = to_2tuple(embed_config.get('data_size', None))
        self.patch_size = to_2tuple(embed_config.get('patch_size', None))

        self.p_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2]) # p1, p2, p3
        self.p_num = self.p_size[0] * self.p_size[1] * self.p_size[2] # p1 * p2 * p3
        self.n_size = (self.data_size[0] // self.patch_size[0],
                       self.data_size[1] // self.patch_size[1],
                       self.data_size[2] // self.patch_size[2]) # n1, n2, n3
        self.n_num = self.n_size[0] * self.n_size[1] * self.n_size[2] # n1 * n2 * n3

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else latent_dim * 2
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        self.if_mask = if_mask
        self.if_remove_cls_embed = if_remove_cls_embed
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_mu = mask_ratio_mu
        self.mask_ratio_std = mask_ratio_std

        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        self.to_latent = nn.Linear(input_dim, latent_dim)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.n_size[1] * self.n_size[2], latent_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.n_size[0], latent_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, latent_dim))
        else:
            if self.cls_embed:
                _num_patches = self.n_num + 1
            else:
                _num_patches = self.n_num

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, latent_dim),
            )

        self.blocks = nn.ModuleList(
            [
                Block(in_features=latent_dim,
                      hidden_features=latent_dim,
                      out_features=latent_dim,
                      norm_layer=norm_layer)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(latent_dim)

        self.final_layer = nn.Linear(latent_dim, self.output_dim)

        self.initialize_weights()

    def random_masking(self, sample, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = sample.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=sample.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        # sort keep ids
        ids_keep = ids_shuffle[:, :len_keep].sort()[0]
        # sort not keep ids
        ids_nokeep = ids_shuffle[:, len_keep:].sort()[0]
        # concat keep ids and not keep ids
        ids_shuffle = torch.concat([ids_keep, ids_nokeep], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        sample_masked = torch.gather(sample, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=sample.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sample_masked, mask, ids_restore, ids_keep


    def forward(self, sample: torch.FloatTensor, mask: torch.LongTensor = None):

        sample = self.to_latent(sample) # (batch_size, num_timesteps, latent_dim)

        N, T, L, C = sample.shape

        sample = sample.reshape(N, T * L, C) # (batch_size, num_sequence, num_features)

        # masking: length -> length * mask_ratio
        if self.if_mask:
            mask_ratio = self.mask_ratio_generator.rvs(1)[0]
            sample, mask, ids_restore, ids_keep = self.random_masking(sample, mask_ratio)
        else:
            mask_ratio = .0
            mask = torch.zeros([N, T * L], device=sample.device)
            ids_restore = torch.arange(T * L, device=sample.device).repeat(N, 1)
            ids_keep = torch.arange(T * L, device=sample.device).repeat(N, 1)

        self.mask_ratio = mask_ratio

        sample = sample.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(sample.shape[0], -1, -1)
            sample = torch.cat((cls_tokens, sample), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.n_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.n_size[1] * self.n_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(sample.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(sample.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(sample.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        sample = sample.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            sample = blk(sample)
        sample = self.norm(sample)

        if self.cls_embed and self.if_remove_cls_embed:
            # remove cls token
            sample = sample[:, 1:, :]
        else:
            sample = sample[:, :, :]

        sample = self.final_layer(sample)

        return sample, mask, ids_restore

if __name__ == '__main__':
    device = torch.device("cpu")

    embed_config = dict(
            type='PatchEmbed',
            data_size=(64, 29, 152),
            patch_size=(4, 1, 152),
            input_channel=1,
            input_dim=152,
            output_dim=128,
            temporal_dim=3,
        )


    encoder = VAELSTMEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim= 128 * 2,
        depth = 2,
        num_heads = 4,
        mlp_ratio = 4.0,
        cls_embed = True,
        sep_pos_embed = True,
        trunc_init = False,
        no_qkv_bias = False,
        if_mask = False,
        mask_ratio_min = 0.5,
        mask_ratio_max = 1.0,
        mask_ratio_mu = 0.55,
        mask_ratio_std = 0.25,
    ).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    embed_layer = EMBED.build(embed_config)

    embed = embed_layer(batch)

    output, mask, ids_restore = encoder(embed)
    print(output.shape)
    print(mask.shape)
    print(ids_restore.shape)

    moments = output

    posterior = DiagonalGaussianDistribution(moments)
    sample = posterior.sample()
    print(sample.shape)