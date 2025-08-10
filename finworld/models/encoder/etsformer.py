import torch
from torch import nn
from typing import Dict, Optional, Type

from finworld.models.encoder.base import Encoder
from finworld.models.modules.etsformer import EtsformerEncodeBlock as Block
from finworld.registry import ENCODER

@ENCODER.register_module(force=True)
class EtsformerEncoder(Encoder):
    """
    Etsformer encoder without CLS token and without random masking.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 output_dim: int,
                 num_heads: int,
                 output_length: int,
                 k: Optional[int],
                 depth: int = 2,
                 dropout: float = 0.1,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 ):

        super().__init__()

        self.input_dim = input_dim
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
                    output_dim=output_dim,
                    output_length=output_length,
                    k=k,
                    latent_dim=latent_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: torch.Tensor
                Input tensor of shape (batch_size, seq_length, input_dim).
            level: torch.Tensor
                Level tensor of shape (batch_size, seq_length, latent_dim).

        Returns:

        """
        x = self.to_latent(x)

        growths = []
        seasons = []

        res = x
        for block in self.blocks:
            res, level, growth, season = block(res, level)
            growths.append(growth)
            seasons.append(season)

        return level, growths, seasons

if __name__ == '__main__':
    device = torch.device('cpu')

    # Example usage
    batch_size = 4
    seq_len = 64
    input_dim = 128
    latent_dim = 128
    output_dim = 128
    num_heads = 4
    output_length = 32
    k = 10
    dropout = 0.1

    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    level = torch.randn(batch_size, seq_len, input_dim).to(device)

    model = EtsformerEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        output_length=output_length,
        k=k,
        dropout=dropout
    ).to(device)

    level, growths, seasons = model(x, level)
    print("Level shape:", level.shape)  # Expected: (batch_size, seq_len, output_dim)
    print("Growths shape:", [g.shape for g in growths])  # Expected: List of tensors with shape (batch_size, seq_len, latent_dim)
    print("Seasons shape:", [s.shape for s in seasons])  # Expected: List of tensors with shape (batch_size, seq_len, latent_dim)