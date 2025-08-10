import torch
import torch.nn as nn

class Loss(nn.Module):
    """
    Base class for all loss functions.
    """
    def __init__(self,
                 **kwargs):
        super(Loss, self).__init__(**kwargs)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        class_name = self.__class__.__name__
        params_str = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        str = f"{class_name}({params_str})"
        return str

    def __repr__(self):
        return self.__str__()
