import torch
from torch import nn
from typing import Dict, Optional

from finworld.models.encoder.base import Encoder
from finworld.models.modules.autoformer import AutoformerEncodeBlock as Block
from finworld.registry import ENCODER

@ENCODER.register_module(force=True)
class AutoformerEncoder(Encoder):
    """
    Autoformer encoder without CLS token and without random masking.
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
        super(AutoformerEncoder, self).__init__(*args, **kwargs)

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
                attn_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Args
        ----
        x : Tensor of shape [B, L, input_dim]

        Returns
        -------
        Tensor of shape [B, L, output_dim]
        """
        x = self.to_latent(x)  # [B, L, latent_dim]

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)  # [B, L, latent_dim]

        x = self.norm(x)
        x = self.proj(x)       # [B, L, output_dim]
        return x

if __name__ == '__main__':
    device = torch.device('cpu')

    model = AutoformerEncoder(
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

    output = model(batch)

    print(f"Output shape: {output.shape}")  # Should be [B, L, output_dim]
