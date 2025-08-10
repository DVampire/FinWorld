import torch
import torch.nn as nn
from typing import Optional
from timm.models.layers import to_2tuple
from einops import rearrange


from finworld.registry import DECODER
from finworld.registry import EMBED
from finworld.models.decoder.base import Decoder
from finworld.models import GRUBlock as Block
from finworld.models.embed import get_patch_info, unpatchify

@DECODER.register_module(force=True)
class VAELSTMDecoder(Decoder):
    def __init__(self,
                 *args,
                 embed_config: dict = None,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 5,
                 depth: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm,
                 cls_embed: bool = True,
                 sep_pos_embed: bool = True,
                 trunc_init: bool = False,
                 no_qkv_bias: bool = False,
                 **kwargs
                 ):
        super(VAELSTMDecoder, self).__init__()

        self.data_size = to_2tuple(embed_config.get('data_size', None))
        self.patch_size = to_2tuple(embed_config.get('patch_size', None))
        self.input_channel = embed_config.get('input_channel', 1)

        self.p_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2]) # p1, p2, p3
        self.p_num = self.p_size[0] * self.p_size[1] * self.p_size[2] # p1 * p2 * p3
        self.n_size = (self.data_size[0] // self.patch_size[0],
                       self.data_size[1] // self.patch_size[1],
                       self.data_size[2] // self.patch_size[2]) # n1, n2, n3
        self.n_num = self.n_size[0] * self.n_size[1] * self.n_size[2] # n1 * n2 * n3

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        self.to_latent = nn.Linear(input_dim, latent_dim)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

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

        self.recon_layer = nn.Linear(
            latent_dim,
            self.p_size[0] * self.p_size[1] * self.p_size[2] * self.input_channel,
            bias=True,
        )

        self.final_layer = nn.Linear(self.data_size[2], self.output_dim)

        self.initialize_weights()

    def forward(self,
                sample: torch.FloatTensor,
                latent_embeds: Optional[torch.FloatTensor] = None,
                ids_restore: torch.LongTensor = None) -> torch.FloatTensor:

            L = self.n_num

            sample = self.to_latent(sample)

            N, _, D = sample.shape

            # masking: length -> length * mask_ratio
            if L + 0 - sample.shape[1] > 0 and ids_restore is not None:
                mask_tokens = self.mask_token.repeat(N, D + 0 - sample.shape[1], 1)
                sample_ = torch.cat([sample[:, :, :], mask_tokens], dim=1)  # no cls token
                sample_ = sample_.view([N, L, D])
                sample_ = torch.gather(
                    sample_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, sample_.shape[2])
                )  # unshuffle
                sample = sample_.view([N, L, D])
            else:
                sample = sample.view([N, L, D])

            # append cls token
            if self.cls_embed:
                cls_token = self.cls_token
                cls_tokens = cls_token.expand(sample.shape[0], -1, -1)
                sample = torch.cat((cls_tokens, sample), dim=1)

            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.n_size[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.n_size[1] * self.n_size[2],
                    dim=1,
                )

                if self.cls_embed:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed_class.expand(
                                pos_embed.shape[0], -1, -1
                            ),
                            pos_embed,
                        ],
                        1,
                    )
            else:
                pos_embed = self.pos_embed[:, :, :]

            # add pos embed
            sample = sample + pos_embed

            # apply Transformer blocks
            for blk in self.blocks:
                sample = blk(sample)
            sample = self.norm(sample)

            # reconstruct to original shape
            sample = self.recon_layer(sample)

            if self.cls_embed:
                # remove cls token
                sample = sample[:, 1:, :]
            else:
                sample = sample[:, :, :]

            patch_info = get_patch_info(sample_size = (N, self.input_channel, *self.data_size),
                                        patch_size = self.patch_size)
            sample = unpatchify(sample, patch_info)

            # predict output
            sample = self.final_layer(sample)

            return sample


if __name__ == '__main__':
    from finworld.models import VAELSTMEncoder
    from finworld.models import DiagonalGaussianDistribution

    device = torch.device("cpu")

    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(1, 29, 4),
        input_channel=1,
        input_dim=152,
        output_dim=128,
        temporal_dim=3,
    )

    encoder = VAELSTMEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=128 * 2,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    ).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)

    embed_layer = EMBED.build(embed_config).to(device)

    embed = embed_layer(batch)

    output, mask, ids_restore = encoder(embed)

    decoder = VAELSTMDecoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=5,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
    ).to(device)

    moments = output
    posterior = DiagonalGaussianDistribution(moments)

    sample = posterior.sample()
    sample = decoder(sample, ids_restore = ids_restore)
    print(sample.shape)
