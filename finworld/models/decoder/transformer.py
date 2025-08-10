import torch
import torch.nn as nn


from finworld.registry import DECODER
from finworld.registry import EMBED
from finworld.models.decoder.base import Decoder
from finworld.models.modules import TransformerRopeBlock as Block

@DECODER.register_module(force=True)
class TransformerDecoder(Decoder):
    def __init__(self,
                 *args,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 5,
                 depth: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm,
                 cls_embed: bool = True,
                 no_qkv_bias: bool = False,
                 trunc_init: bool = False,
                 if_mask: bool = False,
                 if_remove_cls_embed=True,
                 **kwargs
                 ):
        super(TransformerDecoder, self).__init__()

        self.cls_embed = cls_embed

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.trunc_init = trunc_init

        self.if_mask = if_mask
        self.if_remove_cls_embed = if_remove_cls_embed

        self.to_latent = nn.Linear(input_dim, latent_dim)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.blocks = nn.ModuleList(
            [
                Block(latent_dim,
                      num_heads,
                      mlp_ratio,
                      qkv_bias=not no_qkv_bias,
                      norm_layer=norm_layer)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(latent_dim)
        self.proj = nn.Linear(latent_dim, self.output_dim)

        self.initialize_weights()

    def forward(self,
                sample: torch.FloatTensor,
                ids_restore: torch.LongTensor = None) -> torch.FloatTensor:

        sample = self.to_latent(sample)  # (N, M, D)
        N, M, D = sample.shape

        if self.if_mask and ids_restore is not None:
            L = ids_restore.shape[-1]
            num_mask = L - M
            assert num_mask >= 0, "Mask token count cannot be negative"

            mask_tokens = self.mask_token.repeat(N, num_mask, 1)  # (N, L-M, D)
            sample_ = torch.cat([sample, mask_tokens], dim=1)  # (N, L, D)

            sample = torch.gather(
                sample_,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, D)  # (N, L, D)
            )
        else:
            L = sample.shape[1]
            sample = sample.view(N, L, D)

        if self.cls_embed:
            cls_token = self.cls_token.expand(sample.shape[0], -1, -1)
            sample = torch.cat((cls_token, sample), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            sample = blk(sample)
        sample = self.norm(sample)

        if self.cls_embed and self.if_remove_cls_embed:
            sample = sample[:, 1:, :]

        sample = self.proj(sample)
        return sample


if __name__ == '__main__':
    from finworld.models import TransformerEncoder

    device = torch.device("cpu")

    input_channel = 1

    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel=input_channel,
        input_dim=152,
        latent_dim=128,
        output_dim=128,
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

    feature = torch.randn(4, 64, 29, 152).to(device)
    embed_layer = EMBED.build(embed_config).to(device)
    embed = embed_layer(feature)

    output, mask, ids_restore = encoder(embed)

    decoder = TransformerDecoder(
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
    ).to(device)

    sample = decoder(output, ids_restore = ids_restore)
    print(sample.shape)
