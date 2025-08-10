import torch

from finworld.models.base import Model

class Embed(Model):

    """Base class for embedding modules.

    This class serves as a base for all embedding modules in the FinWorld framework.
    It inherits from the Model class and provides a common interface for embedding operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding module.

        Args:
            x (torch.Tensor): Input tensor to be embedded.

        Returns:
            torch.Tensor: Embedded tensor.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")