import torch
from torch import nn
from typing import Dict, Optional, Type, List

from finworld.models.encoder.base import Encoder
from finworld.models.modules.etsformer import EtsformerDecodeBlock as Block
from finworld.registry import DECODER

@DECODER.register_module(force=True)
class EtsformerDecoder(Encoder):
    """
    Etsformer decoder without CLS token and without random masking.
    """

    def __init__(self,
                 num_heads: int,
                 output_length: int,
                 depth: int = 2,
                 dropout: float = 0.1,
                 ):

        super().__init__()

        self.num_heads = num_heads
        self.output_length = output_length
        self.dropout = dropout

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    num_heads=num_heads,
                    output_length=output_length,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.initialize_weights()

    def forward(self,
                growths: List[torch.Tensor],
                seasons: List[torch.Tensor],
                ) -> torch.Tensor:
        """

        Args:
            growths: List[torch.Tensor]
                List of tensors representing growth components, each of shape (batch_size, seq_length, latent_dim).
            seasons: List[torch.Tensor]
                List of tensors representing seasonal components, each of shape (batch_size, seq_length, latent_dim).
        Returns:

        """
        growth_repr = []
        season_repr = []

        for index, blk in enumerate(self.blocks):
            growth = growths[index]
            season = seasons[index]
            growth_horizon, season_horizon = blk(growth, season)

            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)

        growth = torch.stack(growth_repr, dim=0).sum(dim=0)
        season = torch.stack(season_repr, dim=0).sum(dim=0)

        return growth, season

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

    growths = [torch.randn(batch_size, seq_len, latent_dim).to(device) for _ in range(k)]
    seasons = [torch.randn(batch_size, seq_len, latent_dim).to(device) for _ in range(k)]

    model = EtsformerDecoder(
        num_heads=num_heads,
        output_length=output_length,
        dropout=dropout,
    ).to(device)

    growth, season = model(growths, seasons)
    print("Growth shape:", growth.shape)
    print("Season shape:", season.shape)  # Expected: (batch_size, seq_len, latent_dim)