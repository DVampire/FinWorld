import torch
import torch.nn.functional as F

from finworld.registry import LOSS
from finworld.loss.base import Loss

@LOSS.register_module(force=True)
class MSELoss(Loss):
    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction='mean',
                 **kwargs
                 ):
        super(MSELoss, self).__init__(**kwargs)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                **kwargs
                ) -> torch.Tensor:
        """
        Compute the Mean Squared Error (MSE) loss.
        :param y_true: True values
        :param y_pred: Predicted values
        :return: MAE loss
        """

        assert y_pred.shape == y_true.shape, \
            f"Shape mismatch: pred {y_pred.shape} vs target {y_true.shape}"

        loss = F.mse_loss(y_pred,
                         y_true,
                         reduction=self.reduction)

        loss = loss * self.loss_weight

        return loss

if __name__ == '__main__':
    # Example usage
    y_true = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    mse_loss_fn = MSELoss()
    loss_value = mse_loss_fn(y_true, y_pred)
    print(f"MSE Loss: {loss_value.item()}")