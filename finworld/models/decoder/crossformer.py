import torch
from torch import nn
from typing import Dict, Optional, Tuple, List

from finworld.models.decoder.base import Decoder
from finworld.models.modules.crossformer import CrossformerDecodeBlock as Block
from finworld.registry import DECODER

@DECODER.register_module(force=True)
class CrossformerDecoder(Decoder):
    """
    Crossformer decoder without CLS token and without random masking.
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
        super(CrossformerDecoder, self).__init__(*args, **kwargs)

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
                cross: Optional[List[torch.Tensor]] = None,
                attn_mask: Optional[List[torch.Tensor]] = None,
                cross_attn_mask: Optional[List[torch.Tensor]] = None,
                **kwargs
                ) -> torch.Tensor:
        """
        Args
        ----
        x : Tensor of shape [B, T, N, C]

        Returns
        -------
        Tensor of shape [B, L, output_dim]
        """

        x = self.to_latent(x)  # [B, L, latent_dim]

        x_outs = []
        for index, blk in enumerate(self.blocks):
            x = blk(x, cross[index])
            x = self.norms[index](x)
            x = self.projs[index](x)
            x_outs.append(x)

        x = torch.sum(torch.stack(x_outs, dim=0), dim=0)

        return x



if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    num_stocks = 5
    encode_seg_num = 16
    decode_seg_num = 8
    input_dim = 128
    latent_dim = 128
    output_dim = 128
    window_size = 1
    depth = 2
    factor = 1

    model = CrossformerDecoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        seg_num=decode_seg_num,
        window_size=window_size,
        depth=depth,
        factor=factor
    ).to(device)

    x = torch.randn(batch_size, decode_seg_num, num_stocks, input_dim).to(device)
    cross = [torch.randn(batch_size, encode_seg_num, num_stocks, input_dim).to(device) for _ in range(depth)]
    outputs = model(x, cross=cross)
    print(outputs.shape)
