import torch
import torch.nn.functional as F

from finworld.registry import LOSS
from finworld.loss.base import Loss

@LOSS.register_module(force=True)
class PriceLoss(Loss):
    def __init__(self,
                 loss_weight: float = 1.0,
                 **kwargs):
        super(PriceLoss, self).__init__(**kwargs)
        self.loss_weight = loss_weight

    def forward(self,
                y_pred: torch.Tensor,
                mu: torch.Tensor = None,
                sigma: torch.Tensor = None,
                **kwargs
                )-> torch.Tensor:
        """
        :param y_pred: Predicted values
        :param mu: Mean of the distribution
        :param sigma: Standard deviation of the distribution
        :return: Price loss
        """

        if mu is not None:
            mu = mu.expand_as(y_pred)
        if sigma is not None:
            sigma = sigma.expand_as(y_pred)

        if mu is not None:
            restore_pred = y_pred * sigma + mu
        else:
            restore_pred = y_pred

        # close, high, low, open
        close = restore_pred[..., 0]
        high = restore_pred[..., 1]
        low = restore_pred[..., 2]
        open = restore_pred[..., 3]

        # high >= close, open
        upper_loss = F.relu(torch.max(close, open) - high).mean()

        # low <= close, open
        lower_loss = F.relu(low - torch.min(close, open)).mean()

        loss = upper_loss + lower_loss

        loss = loss * self.loss_weight

        return loss