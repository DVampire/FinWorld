import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Any, Union

from finworld.registry import METRIC
from finworld.metric.utils import fill_invalid_values
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class MAE(Metric):
    """Mean Absolute Error (MAE) metric.

    This class computes the Mean Absolute Error between true and predicted values.
    It handles NaN and infinite values by replacing them with zero.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MAE metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(MAE, self).__init__(**kwargs)

    def __call__(self,
                 y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor],
                 mask: Optional[Union[np.ndarray, torch.Tensor]] = None
                 ) -> Union[float, torch.Tensor]:
        """Compute the MAE between true and predicted values.

        Args:
            y_true (Union[np.ndarray, torch.Tensor]): True values.
            y_pred (Union[np.ndarray, torch.Tensor]): Predicted values.
            mask (Optional[Union[np.ndarray, torch.Tensor]]): Optional mask to apply to the true and predicted values.
        Returns:
            float: The computed Mean Absolute Error.
        """
        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            # process nan and inf
            y_true = fill_invalid_values(y_true)
            y_pred = fill_invalid_values(y_pred)

            if mask is not None:
                y_true = y_true * (1.0 - mask)
                y_pred = y_pred * (1.0 - mask)

            mae = np.mean(np.abs(y_true - y_pred))

            return float(mae)
        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            # process nan and inf
            y_true = fill_invalid_values(y_true)
            y_pred = fill_invalid_values(y_pred)

            if mask is not None:
                y_true = y_true * (1.0 - mask)
                y_pred = y_pred * (1.0 - mask)

            mae = torch.mean(torch.abs(y_true - y_pred))

            return mae
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    mae_metric = MAE()
    y_true = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.0, np.inf])
    mae_score = mae_metric(y_true, y_pred)
    print(f"MAE Score: {mae_score}")

    y_true_tensor = torch.tensor([1.0, 2.0, 3.0, float('nan'), 5.0])
    y_pred_tensor = torch.tensor([1.5, 2.5, 3.5, 4.0, float('inf')])
    mae_score_tensor = mae_metric(y_true_tensor, y_pred_tensor)
    print(f"MAE Score (Tensor): {mae_score_tensor.item()}")
