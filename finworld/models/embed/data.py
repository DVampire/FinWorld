import torch
import torch.nn as nn
from tensordict import TensorDict

from finworld.models.embed.base import Embed
from finworld.registry import EMBED
from finworld.models.embed.dense import DenseLinearEmbed
from finworld.models.embed.time import TimeSparseEmbed

@EMBED.register_module(force=True)
class DataEmbed(Embed):
    """Data embedding module. Mainly used for tabular data.

    This module embeds categorical features into a higher-dimensional space.
    It is typically used in transformer models to prepare tabular data for attention mechanisms.

    Args:
        input_dim (int): Dimension of the input dense features.
        start_timestamp (str): Start timestamp of the time series data.
        end_timestamp (str): End timestamp of the time series data.
        level (str): Time level for embedding. Options are '1day', '1hour', '1min', '1sec'.
        output_dim (int): Dimension of the output features.
        dropout (float): Dropout rate for the embedding layer.
    """

    def __init__(self, *args,
                 dense_input_dim: int,
                 sparse_input_dim: int,
                 output_dim: int,
                 latent_dim: int = None,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 level: str = '1day',
                 dropout: float = 0.0,
                 **kwargs
                 ):
        super(DataEmbed, self).__init__(*args, **kwargs)

        self.dense_input_dim = dense_input_dim
        self.sparse_input_dim = sparse_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        if latent_dim is not None:
            self.dense_embed = nn.Sequential(
                DenseLinearEmbed(
                    input_dim=dense_input_dim,
                    output_dim=latent_dim,
                    dropout=dropout
                ),
                nn.ReLU(),
                nn.Linear(latent_dim, output_dim, bias=False)
            )

            self.sparse_embed = nn.Sequential(
                TimeSparseEmbed(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    level=level,
                    output_dim=latent_dim,
                    dropout=dropout
                ),
                nn.ReLU(),
                nn.Linear(latent_dim, output_dim, bias=False)
            )
        else:
            self.dense_embed = DenseLinearEmbed(
                input_dim=dense_input_dim,
                output_dim=output_dim,
                dropout=dropout
            )

            self.sparse_embed = TimeSparseEmbed(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                level=level,
                output_dim=output_dim,
                dropout=dropout
            )

        self.embed = nn.ModuleDict({
            'dense': self.dense_embed,
            'sparse': self.sparse_embed
        })

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self,
                x: TensorDict,
                **kwargs):
        """
        Forward pass for the DataEmbed module.

        Args:
            x (TensorDict): Input tensor dictionary containing 'dense' and 'sparse' keys.
                - 'dense': Tensor of dense features.
                - 'sparse': Tensor of sparse time features.

        Returns:
            torch.Tensor: Embedded features.
        """
        dense_x = x['dense']
        sparse_x = x['sparse']

        dense_embed = self.dense_embed(dense_x)
        sparse_embed = self.sparse_embed(sparse_x)

        # Combine dense and sparse embeddings
        x = dense_embed + sparse_embed

        x = self.dropout(x)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    seq_len = 10

    dense_features = torch.randn(batch_size, seq_len, 64)  # Batch size of 4, 10 time steps, 64 features

    years = torch.randint(2015, 2026, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(1, 8, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature

    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)

    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device)
    }, batch_size=(batch_size, seq_len)).to(device)

    model = DataEmbed(
        dense_input_dim=64,
        sparse_input_dim=4,  # Number of sparse features (days, months, weekdays, years)
        output_dim=64,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        dropout=0.1
    ).to(device)

    output = model(x)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, output_dim)