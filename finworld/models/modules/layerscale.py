import torch
import torch.nn as nn

class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma