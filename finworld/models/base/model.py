import torch
from torch import nn

class Model(nn.Module):
    """Base class for all models in the FinWorld library.

    This class provides a common interface and basic functionality for all models.
    It should be inherited by all model classes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the base model."""
        super().__init__(*args, **kwargs)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, *args, **kwargs):
        """Forward pass of the model. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")