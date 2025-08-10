import torch
from torch import nn
import scipy.stats as stats

from finworld.registry import ENCODER, EMBED
from finworld.models.encoder.base import Encoder
from finworld.models.modules import TransformerRopeBlock as Block
from finworld.models.modules import DiagonalGaussianDistribution

@ENCODER.register_module(force=True)
class TransformerEncoder(Encoder):
    def __init__(self,
                 *args,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 256,
                 depth: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 norm_layer = nn.LayerNorm,
                 cls_embed: bool = True,
                 no_qkv_bias: bool = False,
                 trunc_init: bool = False,
                 if_mask: bool = False,
                 if_remove_cls_embed = True,
                 mask_ratio_min: float = 0.5,
                 mask_ratio_max: float = 1.0,
                 mask_ratio_mu: float = 0.55,
                 mask_ratio_std: float = 0.25,
                 **kwargs
                 ):
        super(TransformerEncoder, self).__init__()

        self.cls_embed = cls_embed

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else latent_dim * 2
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.trunc_init = trunc_init

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

        self.blocks = nn.ModuleList(
            [
                Block(latent_dim,
                      num_heads,
                      mlp_ratio,
                      qkv_bias=not no_qkv_bias,
                      norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(latent_dim)
        self.proj = nn.Linear(latent_dim, self.output_dim)

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
        sample = self.to_latent(sample)

        N, L, C = sample.shape

        sample = sample.reshape(N, L, C)

        if self.if_mask:
            mask_ratio = self.mask_ratio_generator.rvs(1)[0]
            sample, mask, ids_restore, ids_keep = self.random_masking(sample, mask_ratio)
        else:
            mask_ratio = 0.0
            mask = torch.zeros([N, L], device=sample.device)
            ids_restore = torch.arange(L, device=sample.device).repeat(N, 1)
            ids_keep = torch.arange(L, device=sample.device).repeat(N, 1)

        self.mask_ratio = mask_ratio

        sample = sample.view(N, -1, C)

        if self.cls_embed:
            cls_token = self.cls_token.expand(sample.shape[0], -1, -1)
            sample = torch.cat((cls_token, sample), dim=1)

        for blk in self.blocks:
            sample = blk(sample)
        sample = self.norm(sample)

        if self.cls_embed and self.if_remove_cls_embed:
            sample = sample[:, 1:, :]

        sample = self.proj(sample)
        return sample, mask, ids_restore

if __name__ == '__main__':
    device = torch.device("cpu")

    # For 3D
    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel=1,
        input_dim=152,
        output_dim=128,
    )

    encoder = TransformerEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    ).to(device)

    feature = torch.randn(4, 1, 64, 29, 152).to(device)
    embed_layer = EMBED.build(embed_config)
    embed = embed_layer(feature)
    print(embed.shape)
    output, mask, ids_restore = encoder(embed)
    print(output.shape)
    print(mask.shape)
    print(ids_restore.shape)
    posterior = DiagonalGaussianDistribution(output)
    sample = posterior.sample()
    print(sample.shape)

    # For 2D
    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 152),
        patch_size=(4, 152),
        input_channel=1,
        input_dim=152,
        output_dim=128,
        temporal_dim=3,
    )

    encoder = TransformerEncoder(
        input_dim=128,
        latent_dim=128,
        output_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    ).to(device)

    feature = torch.randn(4, 1, 64, 152).to(device)
    embed_layer = EMBED.build(embed_config)
    embed = embed_layer(feature)
    print(embed.shape)
    output, mask, ids_restore = encoder(embed)
    print(output.shape)
    print(mask.shape)
    print(ids_restore.shape)
