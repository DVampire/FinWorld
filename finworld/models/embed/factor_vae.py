import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from einops import rearrange

from finworld.registry import EMBED
from finworld.models.modules.transformer import Mlp


@EMBED.register_module(force=True)
class FactorVAEEmbed(nn.Module):
    """Image to FactorVAE Embedding"""

    def __init__(
        self,
        *args,
        data_size=(64, 29, 152),
        patch_size=(64, 1, 152),
        input_channel: int = 1,
        input_dim: int = 152,
        temporal_dim: int = 3,
        embed_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        data_size = to_2tuple(data_size)
        patch_size = to_2tuple(patch_size)
        assert (data_size[0] % patch_size[0] == 0 and data_size[1] % patch_size[1] == 0
                and data_size[2] % patch_size[2] == 0), f"Data size {self.data_size} must be divisible by patch size {self.patch_size}"

        self.input_channel = input_channel
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        self.data_size = data_size
        self.patch_size = patch_size

        self.input_size = (self.data_size[0] // self.patch_size[0], self.data_size[1] // self.patch_size[1], self.data_size[2] // self.patch_size[2])
        self.num_patches = self.input_size[0] * self.input_size[1] * self.input_size[2]

        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            act_layer=nn.LeakyReLU
        )

        self.rnn = nn.GRU(input_size=self.embed_dim,
                          hidden_size=self.embed_dim,
                          batch_first=True)

    def forward(self, sample):
        N, C, T, S, F = sample.shape # batch, channel, temporal, spatial, feature
        assert (T == self.data_size[0] and S == self.data_size[1]
                and F == self.data_size[2]), f"Input data size {(T, N, F)} doesn't match model {self.data_size}."

        sample = rearrange(sample, 'n c t s f -> (n c s) t f')

        sample = self.mlp(sample)

        out, hidden = self.rnn(sample)
        hidden = hidden.squeeze(0)
        e = rearrange(hidden, '(n c s) f -> (n c) s f', n = N, c = C, s = S)

        return e


if __name__ == '__main__':
    device = torch.device("cpu")

    model = FactorVAEEmbed(data_size=(64, 29, 152),
        patch_size=(64, 1, 152),
        input_dim = 152,
        input_channel=1,
        temporal_dim= 3,
        embed_dim= 128).to(device)
    print(model)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    emb = model(batch)
    print(emb.shape)