import torch
import torch.nn as nn

from finworld.models.embed.base import Embed
from finworld.registry import EMBED

@EMBED.register_module(force=True)
class AggEmbed(Embed):
    """Aggregation embedding module. This module is used to aggregate features from multiple sources into a single representation.

    It can be used in transformer models to prepare aggregated data for attention mechanisms.
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        latent_dim (int, optional): Dimension of the latent space. If None, input_dim is used.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for the embedding layer.
    """
    def __init__(self,
                 *args,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int = None,
                 num_heads: int = 4,
                 dropout=0.0,
                 **kwargs):

        super(AggEmbed, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        if latent_dim is not None:

            self.stem = nn.Sequential(
                nn.Linear(input_dim, latent_dim, bias=False),
                nn.ReLU(),
            )

            self.q_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # Pooling to get a single vector per sequence
                nn.Flatten(),
                nn.Linear(latent_dim, latent_dim, bias=False),
                nn.ReLU(),
            )

            self.attn = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout
            )

            self.proj = nn.Linear(latent_dim, output_dim, bias=False)
        else:
            self.stem = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=False),
                nn.ReLU(),
            )

            self.q_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(output_dim, output_dim, bias=False),
                nn.ReLU(),
            )

            self.attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout
            )

            self.proj = nn.Linear(output_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.initialize_weights()

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the embedding layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_asset, input_features).
                - batch_size: Number of samples in the batch.
                - seq_len: Length of the sequence (number of time steps).
                - num_asset: Number of assets.
                - input_features: Dimension of the input features.

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, output_dim).
        """

        B, T, N, C = x.shape

        x = x.reshape(B * T, N, C) # Reshape to (B*T, N, C) for aggregation
        x = self.stem(x)

        x_tp = x.transpose(1, 2) # Transpose to (B*T, C, N) for pooling
        q = self.q_proj(x_tp)
        q = q.unsqueeze(1) # Add sequence dimension for attention, resulting in (B*T, 1, C)

        attn_output, _ = self.attn(q, x, x) # Apply attention, output shape (B*T, 1, C)

        x = self.proj(attn_output).reshape(B, T, -1) # Project to output dimension and reshape back to (B, T, output_dim)

        x = self.dropout(x)

        return x


if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    seq_len = 10
    num_asset = 5
    input_dim = 64
    latent_dim = 32
    output_dim = 32

    x = torch.randn(batch_size, seq_len, num_asset, input_dim).to(device)

    agg_embed = AggEmbed(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        num_heads=4,
        dropout=0.0
    ).to(device)

    output = agg_embed(x)
    print(output.shape)  # Expected shape: (batch_size, seq_len, output_dim)
