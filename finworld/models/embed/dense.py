import torch
import torch.nn as nn

from finworld.models.embed.base import Embed
from finworld.registry import EMBED

@EMBED.register_module(force=True)
class DenseLinearEmbed(Embed):
    """Value embedding module. Mainly used for dense input data.

    This module embeds input values into a higher-dimensional space.
    It is typically used in transformer models to prepare input data for attention mechanisms.

    Args:
        input_dim (int): Dimension of the input features.
        latent_dim (int): Dimension of the latent space.
        output_dim (int): Dimension of the output features.
    """

    def __init__(self,
                 *args,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int = None,
                 dropout=0.0,
                 **kwargs):

        super(DenseLinearEmbed, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        if latent_dim is not None:
            self.embed = nn.Sequential(
                nn.Linear(input_dim, latent_dim, bias=False),
                nn.ReLU(),
                nn.Linear(latent_dim, output_dim, bias=False)
            )
        else:
            self.embed = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.initialize_weights()

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, output_dim).
        """
        x = self.embed(x)
        x = self.dropout(x)

        return x

@EMBED.register_module(force=True)
class DenseConv1dEmbed(Embed):
    """Value embedding module using 1D convolution. Mainly used for dense input data.

    This module embeds input values into a higher-dimensional space using a 1D convolutional layer.
    It is typically used in transformer models to prepare input data for attention mechanisms.

    Args:
        input_dim (int): Dimension of the input features.
        embed_dim (int, optional): Dimension of the output features. Defaults to None.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
    """

    def __init__(self,
                 *args,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int = None,
                 kernel_size=3,
                 padding=1,
                 dropout=0.0,
                 **kwargs):
        super(DenseConv1dEmbed, self).__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        if latent_dim is not None:
            self.embed = nn.Sequential(
                nn.Conv1d(input_dim,
                          latent_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          padding_mode='circular',
                          bias=False),
                nn.ReLU(),
                nn.Conv1d(latent_dim,
                          output_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          padding_mode='circular',
                         bias=False)
            )
        else:
            self.embed = nn.Conv1d(input_dim,
                                   output_dim,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.initialize_weights()

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, output_dim).
        """
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    device = torch.device("cpu")

    batch = torch.randn(2, 10, 64).to(device)  # Example batch of shape (batch_size, seq_len, input_dim)
    model = DenseLinearEmbed(input_dim=64, output_dim=64).to(device)
    print(model)
    output = model(batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)

    model = DenseConv1dEmbed(input_dim=64, output_dim=64, kernel_size=3, padding=1).to(device)
    print(model)
    output = model(batch)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_len, output_dim)
