import torch
from torch import nn
from typing import Dict, Optional, Tuple

from finworld.models.decoder.base import Decoder
from finworld.models.modules.autoformer import AutoformerDecodeBlock as Block
from finworld.registry import DECODER

@DECODER.register_module(force=True)
class AutoformerDecoder(Decoder):
    """
    Plain Transformer encoder without CLS token and without random masking.
    """
    def __init__(
        self,
        *args,
        input_dim: int = 128,
        latent_dim: int = 128,
        output_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        no_qkv_bias: bool = False,
        moving_avg: int = 25,
        factor: int = 1,
        **kwargs
    ) -> None:
        super(AutoformerDecoder, self).__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Linear projection from input dimension to latent dimension
        self.to_latent = nn.Linear(input_dim, latent_dim, bias=True)

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                    moving_avg=moving_avg,
                    factor=factor,
                )
                for _ in range(depth)
            ]
        )

        # Final layer norm and projection
        self.norm = norm_layer(latent_dim)
        self.proj = nn.Linear(latent_dim, self.output_dim)

        self.initialize_weights()

    def forward(self,
                x: torch.Tensor,
                cross: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x : Tensor of shape [B, L, input_dim]

        Returns
        -------
        Tensor of shape [B, L, output_dim]
        """
        x = self.to_latent(x)  # [B, L, latent_dim]

        trend = 0
        for blk in self.blocks:
            x, residual_trend = blk(x,
                                    cross = cross,
                                    attn_mask = attn_mask,
                                    cross_attn_mask = cross_attn_mask,
                                    )  # [B, L, latent_dim]
            trend = trend + residual_trend  # accumulate trend components

        x = self.norm(x)
        x = self.proj(x)       # [B, L, output_dim]
        return x, trend

if __name__ == '__main__':
    device = torch.device('cpu')

    model = AutoformerDecoder(
        input_dim=64,
        latent_dim=64,
        output_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        no_qkv_bias=False,
        moving_avg=25,
        factor=1
    ).to(device)

    batch = torch.randn(2, 10, 64).to(device).to(device)  # [B, L, input_dim]

    output, trend = model(batch)

    print(f"Output shape: {output.shape}")  # Should be [B, L, output_dim]
    print(f"Trend shape: {trend.shape}")  # Should be [B, L, latent_dim]
