import torch
import torch.nn as nn
from tensordict import TensorDict

from finworld.models.embed.base import Embed
from finworld.registry import EMBED
from finworld.models.embed.time import TimeDenseEmbed
from finworld.models.embed.dense import DenseLinearEmbed
from finworld.models.embed.sparse import SparseEmbed

@EMBED.register_module(force=True)
class TradingDataEmbed(Embed):
    """Trading Data embedding module.

    This module embeds trading trajectory data into a higher-dimensional space.

    Args:
        input_dim (int): Dimension of the input dense features.
        start_timestamp (str): Start timestamp of the time series data.
        end_timestamp (str): End timestamp of the time series data.
        level (str): Time level for embedding. Options are '1day', '1hour', '1min', '1sec'.
        output_dim (int): Dimension of the output features.
        dropout (float): Dropout rate for the embedding layer.
    """

    def __init__(self, 
                 *args,
                 dense_input_dim: int,
                 sparse_input_dim: int,
                 output_dim: int,
                 latent_dim: int = None,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 level: str = '1day',
                 dropout: float = 0.0,
                 num_heads: int = 4,
                 num_max_tokens: int = 1000,
                 if_use_trajectory: bool = False,
                 if_use_sparse: bool = False,
                 **kwargs
                 ):
        super(TradingDataEmbed, self).__init__(*args, **kwargs)

        self.dense_input_dim = dense_input_dim
        self.sparse_input_dim = sparse_input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.if_use_trajectory = if_use_trajectory
        self.if_use_sparse = if_use_sparse

        if latent_dim is not None:

            if self.if_use_sparse:

                self.dense_embed = DenseLinearEmbed(
                        input_dim=dense_input_dim,
                        output_dim=latent_dim,
                        dropout=dropout
                    )

                self.sparse_embed = TimeDenseEmbed(
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        level=level,
                        output_dim=latent_dim,
                        dropout=dropout
                    )
            else:
                self.dense_embed = DenseLinearEmbed(
                        input_dim=dense_input_dim + sparse_input_dim,
                        output_dim=latent_dim,
                        dropout=dropout
                    )

            if self.if_use_trajectory:
            
                self.cashes_embed = DenseLinearEmbed(
                        input_dim=1,
                        output_dim=latent_dim // 4,
                        dropout=dropout
                    )

                self.positions_embed = SparseEmbed(
                    num_embeddings=num_max_tokens,
                    output_dim=latent_dim // 4,
                    dropout=dropout
                )

                self.rets_embed = DenseLinearEmbed(
                    input_dim=1,
                    output_dim=latent_dim // 4,
                    dropout=dropout
                )

                self.actions_embed = SparseEmbed(
                    num_embeddings=3,
                    output_dim=latent_dim // 4,
                    dropout=dropout
                )
            
        else:
            if self.if_use_sparse:
                self.dense_embed = DenseLinearEmbed(
                        input_dim=dense_input_dim,
                        output_dim=output_dim,
                        dropout=dropout
                    )

                self.sparse_embed = TimeDenseEmbed(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    level=level,
                    output_dim=output_dim,
                    dropout=dropout
                )
            else:
                self.dense_embed = DenseLinearEmbed(
                    input_dim=dense_input_dim + sparse_input_dim,
                    output_dim=output_dim,
                    dropout=dropout
                )

            if self.if_use_trajectory:
                self.cashes_embed = DenseLinearEmbed(
                    input_dim=1,
                    output_dim=output_dim // 4,
                    dropout=dropout
                )

                self.positions_embed = DenseLinearEmbed(
                    num_embeddings=1,
                    output_dim=output_dim // 4,
                    dropout=dropout
                )

                self.rets_embed = DenseLinearEmbed(
                    input_dim=1,
                    output_dim=output_dim // 4,
                    dropout=dropout
                )

                self.actions_embed = SparseEmbed(
                    num_embeddings=3,
                    output_dim=output_dim // 4,
                    dropout=dropout
                )

        self.embed = nn.ModuleDict({
            'dense': self.dense_embed,
        })

        if self.if_use_sparse:
            self.embed.update({
                'sparse': self.sparse_embed
            })

        if self.if_use_trajectory:
            self.embed.update({
                'cashes': self.cashes_embed,
                'positions': self.positions_embed,
                'rets': self.rets_embed,
                'actions': self.actions_embed
            })

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.proj = nn.Linear(
            in_features=latent_dim if latent_dim is not None else output_dim,
            out_features=output_dim
        ) if latent_dim is not None else nn.Identity()

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

        if self.if_use_sparse:
            dense_embed = self.dense_embed(dense_x)
            sparse_embed = self.sparse_embed(sparse_x)

            base_embed = dense_embed + sparse_embed
        else:
            dense_x = torch.cat([dense_x, sparse_x], dim=-1)
            dense_embed = self.dense_embed(dense_x)
            base_embed = dense_embed

        if self.if_use_trajectory:
            cashes_x = x['cashes']
            positions_x = x['positions']
            rets_x = x['rets']
            actions_x = x['actions']
            cashes_embed = self.cashes_embed(cashes_x.unsqueeze(-1))
            positions_embed = self.positions_embed(positions_x.unsqueeze(-1))
            rets_embed = self.rets_embed(rets_x.unsqueeze(-1))
            actions_embed = self.actions_embed(actions_x.unsqueeze(-1))

            combined_embed = torch.cat([cashes_embed,
                                        positions_embed,
                                        rets_embed,
                                        actions_embed], dim=-1)

            # Combine dense and sparse embeddings
            x = base_embed + combined_embed
        else:
            x = base_embed

        x = self.proj(x)
        x = self.dropout(x)

        return x

if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 2
    seq_len = 10
    input_dim = 64
    latent_dim = 32
    output_dim = 32

    dense_features = torch.randn(batch_size, seq_len, input_dim)  # Batch size of 4, 10 time steps, 64 features

    years = torch.randint(2015, 2026, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature

    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)

    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device),
        'cashes': torch.randn(batch_size, seq_len).to(device),
        'positions': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'rets': torch.randn(batch_size, seq_len).to(device),
        'actions': torch.randint(0, 3, (batch_size, seq_len)).to(device)
    }, batch_size=(batch_size, seq_len)).to(device)

    model = TradingDataEmbed(
        dense_input_dim=input_dim,
        latent_dim=latent_dim,
        sparse_input_dim=4,  # Number of sparse features (days, months, weekdays, years)
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        num_max_tokens=1000,
        dropout=0.1
    ).to(device)

    output = model(x)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, output_dim)