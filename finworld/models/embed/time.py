import numpy as np
import torch
import torch.nn as nn

from finworld.models.embed.base import Embed
from finworld.registry import EMBED
from finworld.models.embed.sparse import SparseEmbed
from finworld.utils import TimeLevel, TimeLevelFormat
from finworld.utils import calculate_time_info
from finworld.models.embed.utils import sin_cos_encode

@EMBED.register_module(force=True)
class TimeDenseEmbed(Embed):
    """Time embedding module. Mainly used for time series data.

    This module embeds time-related features into a higher-dimensional space.
    It is typically used in transformer models to prepare time series data for attention mechanisms.

    Args:
        start_timestamp (str): Start timestamp of the time series data.
        end_timestamp (str): End timestamp of the time series data.
        level (str): Time level for embedding. Options are '1day', '1hour', '1min', '1sec'.
        output_dim (int): Dimension of the output features.
        dropout (float): Dropout rate for the embedding layer.
    """

    def __init__(self, *args,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 level: str = '1day',
                 output_dim: int = 64,
                 dropout: float = 0.0,
                 **kwargs
                 ):

        super(TimeDenseEmbed, self).__init__(*args, **kwargs)

        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.output_dim = output_dim
        self.dropout = dropout

        # We must order the columns to ensure the embed layer is consistent with the input time data
        self.time_info = calculate_time_info(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level,
        )

        self.embedding_info = self.time_info['embedding_info']
        self.ordered_columns = self.time_info['columns']

        input_dim = (len(self.ordered_columns) - 1) * 3 + 1 # the last column is year, which is encoded as a single value
        self.embed = nn.Linear(
            in_features=input_dim,
            out_features=self.output_dim,
        )

        self.dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the time embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
                The features should be in the order defined by self.ordered_columns.

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, output_dim).
        """
        embedded_features = []
        for i, column in enumerate(self.ordered_columns):
            feature = x[..., i]

            if column == 'year':
                # Year is encoded as a single value
                min_value = self.embedding_info[column]['min']
                length = self.embedding_info[column]['length']
                embedded_feature = (feature.unsqueeze(-1) - min_value) / length
            else:
                # Scale the feature to the range of the embedding
                min_value = self.embedding_info[column]['min']
                length = self.embedding_info[column]['length']
                embedded_feature = sin_cos_encode(
                    x=feature,
                    length=length,
                    min_value=min_value,
                )
            embedded_features.append(embedded_feature)

        # Sum up the embedded features
        x = torch.concat(embedded_features, dim=-1)  # Shape: (B, L, input_dim)
        x = self.embed(x)
        x = self.dropout(x)
        return x


@EMBED.register_module(force=True)
class TimeSparseEmbed(Embed):
    """Time embedding module. Mainly used for time series data.

    This module embeds time-related features into a higher-dimensional space.
    It is typically used in transformer models to prepare time series data for attention mechanisms.

    Args:
        start_timestamp (str): Start timestamp of the time series data.
        end_timestamp (str): End timestamp of the time series data.
        level (str): Time level for embedding. Options are '1day', '1hour', '1min', '1sec'.
        output_dim (int): Dimension of the output features.
        dropout (float): Dropout rate for the embedding layer.
    """

    def __init__(self, *args,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 level: str = '1day',
                 output_dim: int = 64,
                 dropout: float = 0.0,
                 **kwargs
                 ):

        super(TimeSparseEmbed, self).__init__(*args, **kwargs)

        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.output_dim = output_dim
        self.dropout = dropout

        # We must order the columns to ensure the embed layer is consistent with the input time data
        self.time_info = calculate_time_info(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level,
        )

        self.embedding_info = self.time_info['embedding_info']
        self.ordered_columns = self.time_info['columns']

        self.embed = nn.ModuleDict()
        for name, item in self.embedding_info.items():
            num_embeddings = item['length']
            embed_layer = SparseEmbed(
                num_embeddings=num_embeddings,
                output_dim=self.output_dim,
            )
            self.embed[name] = embed_layer

        self.dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the time embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
                The features should be in the order defined by self.ordered_columns.

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, output_dim).
        """
        embedded_features = []
        for i, column in enumerate(self.ordered_columns):
            feature = x[..., i]

            # Scale the feature to the range of the embedding
            min_value = self.embedding_info[column]['min']
            feature = feature - min_value

            embedded_feature = self.embed[column](x=feature)
            embedded_features.append(embedded_feature)

        # Sum up the embedded features
        x = torch.stack(embedded_features, dim=-1)

        x = x.sum(dim=-1)  # Shape: (B, L, output_dim)

        x = self.dropout(x)

        return x

if __name__ == '__main__':
    device = torch.device("cpu")

    # Example usage
    start_timestamp = "2015-05-01"
    end_timestamp = "2025-05-01"
    level = "1day"
    output_dim = 64

    model = TimeSparseEmbed(start_timestamp=start_timestamp,
                      end_timestamp=end_timestamp,
                      level=level,
                      output_dim=output_dim).to(device)

    # Create a dummy input tensor with shape (batch_size, seq_len, num_features)
    batch_size = 2
    seq_len = 10

    years = torch.randint(0, 10, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature
    batch = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)

    output = model(batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)

    model = TimeDenseEmbed(start_timestamp=start_timestamp,
                           end_timestamp=end_timestamp,
                           level=level,
                           output_dim=output_dim).to(device)
    output = model(batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)