import torch
from torch import nn
from typing import Dict, Optional

from finworld.models.encoder.base import Encoder
from finworld.models.modules.crossformer import CrossformerEncodeBlock as Block
from finworld.registry import ENCODER

@ENCODER.register_module(force=True)
class CrossformerEncoder(Encoder):
    """
    Crossformer encoder without CLS token and without random masking.
    """

    def __init__(
        self,
        *args,
        input_dim: int = 128,
        latent_dim: int = 128,
        output_dim: int = 128,
        seg_num: int = 16,
        window_size: int = 1,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        no_qkv_bias: bool = False,
        factor: int = 1,
        **kwargs
    ) -> None:
        super(CrossformerEncoder, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Linear projection from input dimension to latent dimension
        self.to_latent = nn.Linear(input_dim, latent_dim, bias=False)

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=latent_dim,
                    num_heads=num_heads,
                    seg_num=seg_num,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                    factor=factor,
                )
                for _ in range(depth)
            ]
        )

        # Final layer norm and projection
        self.norms = nn.ModuleList(
            [
                norm_layer(latent_dim) for _ in range(depth)
            ]
        )
        self.projs = nn.ModuleList(
            [
                nn.Linear(latent_dim, output_dim) for _ in range(depth)
            ]
        )

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

        x_outs = []
        for index, blk in enumerate(self.blocks):
            x = blk(x)  # [B, L, latent_dim]
            x = self.norms[index](x)  # Normalize the output of the block
            x = self.projs[index](x) # Project to output dimension
            x_outs.append(x)

        return x_outs

if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    num_stocks = 5
    seg_num = 10
    input_dim = 128
    latent_dim = 128
    output_dim = 128
    window_size = 1
    depth = 2
    factor = 1

    model = CrossformerEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        seg_num=seg_num,
        window_size=window_size,
        depth=depth,
        factor=factor
    ).to(device)

    x = torch.randn(batch_size, seg_num, num_stocks, input_dim).to(device)
    outputs = model(x)

    for i, out in enumerate(outputs):
        print(f"Output from block {i+1}: {out.shape}")

