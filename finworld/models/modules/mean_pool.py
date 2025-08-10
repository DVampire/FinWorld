import torch
import torch.nn as nn

class MeanPool1D(nn.Module):
    """ Mean pooling over the last dimension of the input tensor.
    """

    def __init__(self, dim: int = -1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)