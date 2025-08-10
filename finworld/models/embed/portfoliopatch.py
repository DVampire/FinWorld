import torch
import torch.nn as nn
from tensordict import TensorDict

from finworld.models.embed.base import Embed
from finworld.registry import EMBED
from finworld.models.embed.utils import sin_cos_encode
from finworld.utils import calculate_time_info
from finworld.utils import TimeLevel
from finworld.models.embed.patch import PatchEmbed

@EMBED.register_module(force=True)
class PortfolioPatchEmbed(Embed):
    def __init__(self,
                 *args,
                 dense_input_dim: int,
                 sparse_input_dim: int,
                 num_assets: int,
                 output_dim: int,
                 latent_dim: int = None,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 history_timestamps: int = 64,
                 patch_timestamps: int = 4,
                 dropout: float = 0.0,
                 if_use_trajectory: bool = False,
                 if_use_sparse: bool = False,
                 level: str = '1day',
                 **kwargs
                 ):
        super(PortfolioPatchEmbed, self).__init__(*args, **kwargs)

        self.level = TimeLevel.from_string(level)
        self.dense_input_dim = dense_input_dim
        self.sparse_input_dim = sparse_input_dim
        self.num_assets = num_assets
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.if_use_trajectory = if_use_trajectory
        self.if_use_sparse = if_use_sparse
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        data_size = [history_timestamps, num_assets, dense_input_dim]
        patch_size = [patch_timestamps, 1, dense_input_dim]

        self.time_info = calculate_time_info(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level,
        )

        self.embedding_info = self.time_info['embedding_info']
        self.ordered_columns = self.time_info['columns']

        if self.if_use_sparse:
            sparse_extra_dim = (self.sparse_input_dim - 1) * 3 + 1 # days, months, weekdays, years
            data_size = [data_size[0], data_size[1], data_size[2] + sparse_extra_dim]
            patch_size = [patch_size[0], patch_size[1], patch_size[2] + sparse_extra_dim]
        if self.if_use_trajectory:
            raise NotImplementedError("Trajectory embedding is not implemented for PortfolioPatchEmbed.")

        self.data_size = tuple(data_size)
        self.patch_size = tuple(patch_size)

        self.embed = PatchEmbed(
            data_size=data_size,
            patch_size=patch_size,
            input_dim=data_size[-1],
            latent_dim=latent_dim,
            output_dim=output_dim,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.proj = nn.Linear(output_dim, output_dim)

        self.initialize_weights()

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

        embedded_features = [dense_x]

        if self.if_use_sparse:
            sparse_x = sparse_x.unsqueeze(-2).repeat_interleave(dense_x.shape[-2], dim=-2)
            for i, column in enumerate(self.ordered_columns):
                feature = sparse_x[..., i]

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

        if self.if_use_trajectory:
            raise NotImplementedError("Trajectory embedding is not implemented for PortfolioPatchEmbed.")

        x = torch.cat(embedded_features, dim=-1)

        x = self.embed(x)
        x = self.proj(x)
        x = self.dropout(x)

        return x

if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    seq_len = 64
    num_assets = 6
    input_dim = 64
    latent_dim = 32
    output_dim = 32

    dense_features = torch.randn(batch_size, seq_len, num_assets, input_dim)  # Batch size of 4, 10 time steps, 64 features

    years = torch.randint(0, 10, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature

    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(
        device)  # Shape: (batch_size, seq_len, num_features)

    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device),
        'cashes': torch.randn(batch_size, seq_len).to(device),
        'positions': torch.randn(batch_size, seq_len).to(device),
        'rets': torch.randn(batch_size, seq_len).to(device),
        'actions': torch.randint(0, 3, (batch_size, seq_len)).to(device)
    }, batch_size=(batch_size, seq_len)).to(device)

    model = PortfolioPatchEmbed(
        dense_input_dim=input_dim,
        sparse_input_dim=4,  # Number of sparse features (days, months, weekdays, years)
        num_assets= num_assets,
        latent_dim=latent_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        patch_timestamps=4,
        history_timestamps=seq_len,
        level='1day',
        if_use_sparse=True,
        if_use_trajectory=False,
        dropout=0.1
    ).to(device)

    output = model(x)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, output_dim)