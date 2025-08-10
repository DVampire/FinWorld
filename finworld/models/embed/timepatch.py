import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

from finworld.registry import EMBED
from finworld.models.embed.utils import get_patch_info

@EMBED.register_module(force=True)
class TimePatchEmbed(nn.Module):
    """TimePatchEmbed is a module that performs patch embedding on input data."""

    def __init__(
        self,
        *args,
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel: int = 1,
        input_dim: int = 152,
        latent_dim: int = 128,
        output_dim: int = 128,
        if_use_stem: bool = False,
        stem_embedding_dim: int = 64,
        **kwargs
    ):
        super(TimePatchEmbed, self).__init__()

        self.data_size, self.patch_size = self._check_input(data_size, patch_size)

        self.input_channel = input_channel
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else latent_dim

        self.patch_info = get_patch_info(
            input_channel=self.input_channel,
            data_size=self.data_size,
            patch_size=self.patch_size
        )

        self.if_use_stem = if_use_stem
        self.stem_embedding_dim = stem_embedding_dim

        if self.if_use_stem:
            self.stem_layer = nn.Linear(self.data_size[-1], self.stem_embedding_dim)
            kernel_size = tuple(self.patch_size[:-1]) + (self.stem_embedding_dim,)
        else:
            kernel_size = tuple(self.patch_size)

        if len(kernel_size) == 2:
            self.conv = nn.Conv2d(in_channels=self.input_channel,
                                  out_channels=self.latent_dim,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        elif len(kernel_size) == 3:
            self.conv = nn.Conv3d(in_channels=self.input_channel,
                                  out_channels=self.latent_dim,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        else:
            raise ValueError(f"Unsupported kernel size {kernel_size}. Only 2D and 3D are supported.")

        self.proj = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.output_dim
        )
        # Initialize the convolutional layer weights
        torch.nn.init.xavier_uniform_(self.conv.weight.data.view([self.proj.weight.shape[0], -1]))

    def _check_input(self, data_size, patch_size):
        data_size = to_2tuple(data_size)
        patch_size = to_2tuple(patch_size)

        assert len(data_size) == len(patch_size), f"Data size {data_size} and patch size {patch_size} must have the same length."
        assert all(ds % ps == 0 for ds, ps in zip(data_size, patch_size)), \
            f"Data size {data_size} must be divisible by patch size {patch_size}"

        return data_size, patch_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, D1, D2, ..., Dn), where:
               - N is the batch size
               - C is the input channel
               - D1...Dn are spatial/temporal/feature dimensions

        Returns:
            Tensor of shape (N, Num_Patches, C_out) after patch embedding
        """

        # 1. Check input dimensionality and consistency
        input_shape = x.shape  # (N, D1, D2, ..., Dn)
        spatial_shape = input_shape[1:]  # (D1, ..., Dn)

        x = x.unsqueeze(1) # Add channel dimension if missing, (N, C, D1, D2, ..., Dn)

        assert len(spatial_shape) == len(self.data_size), \
            f"Input dims {spatial_shape} do not match model data dims {self.data_size}"

        for i in range(len(self.data_size)):
            assert spatial_shape[i] == self.data_size[i], \
                f"Dimension {i} mismatch: got {spatial_shape[i]}, expected {self.data_size[i]}"

        # 2. Apply stem layer (if enabled) to the last dimension, e.g. feature dimension
        if self.if_use_stem:
            # Collapse all but the last dimension and apply linear projection
            *prefix_dims, feature_dim = x.shape
            x = self.stem_layer(x)  # Applies nn.Linear(feature_dim → stem_embedding_dim)
            x = x.view(*prefix_dims, self.stem_embedding_dim)

        # 3. Apply projection layer (patch embedding via ConvXd)
        x = self.conv(x)  # Shape becomes (N, embed_dim, D1', D2', ...)

        # 4. Flatten spatial dimensions (keeping batch and channel separate)
        x = x.squeeze(-1)  # Remove the last channel dimension 1, now (N, C_out, D1', D2', ...)
        N, C_out, *spatial = x.shape
        x = x.movedim(1, -1)

        # 5. Apply final projection to output dimension
        x = self.proj(x)  # (N, D1', D2', ..., output_dim)

        return x


if __name__ == '__main__':
    device = torch.device("cpu")

    model = TimePatchEmbed(data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_dim = 152,
        input_channel=1,
        embed_dim= 128,
        if_use_stem=True,
        stem_embedding_dim=64).to(device)
    print(model)
    batch = torch.randn(4, 64, 29, 152).to(device)
    emb = model(batch)
    print(emb.shape)

    model = TimePatchEmbed(data_size=(64, 152),
                       patch_size=(4, 152),
                       input_dim=152,
                       input_channel=1,
                       embed_dim=128,
                       if_use_stem=True,
                       stem_embedding_dim=64).to(device)
    print(model)
    batch = torch.randn(4, 64, 152).to(device)
    emb = model(batch)
    print(emb.shape)