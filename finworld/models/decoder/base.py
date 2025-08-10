import torch
import torch.nn as nn

from finworld.models.modules.transformer import TransformerBlock as Block

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 128,
        latent_dim: int = 128,
        output_dim: int = 5,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        no_qkv_bias: bool = False,
        trunc_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.trunc_init = trunc_init

        # Linear projection from input space to latent space
        self.to_latent = nn.Linear(input_dim, latent_dim, bias=True)

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        # Final layer norm and projection
        self.norm = norm_layer(latent_dim)
        self.proj = nn.Linear(latent_dim, self.output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # --------------------------------------------------------------------- #
    #                               forward                                 #
    # --------------------------------------------------------------------- #
    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args
        ----
        sample : Tensor of shape [B, L, input_dim]

        Returns
        -------
        Tensor of shape [B, L, output_dim]
        """
        x = self.to_latent(sample)  # [B, L, latent_dim]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.proj(x)            # [B, L, output_dim]
        return x
