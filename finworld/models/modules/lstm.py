from functools import partial
from torch import nn as nn
from timm.layers import to_2tuple

class GRUBlock(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 norm_layer=None,
                 bias=True,
                 drop=0.,
                 use_conv=False,):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.gru1 = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_features,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=drop_probs[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.gru2 = nn.GRU(
            input_size=hidden_features,
            hidden_size=out_features,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=drop_probs[1],
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x, _ = self.gru1(x)
        x = self.act(x)
        x = self.drop1(x)
        x, _ = self.gru2(x)
        x = self.drop2(x)
        return x