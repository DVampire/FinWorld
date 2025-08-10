import torch
import numpy as np
from typing import Union

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class HitRatio(Metric):
    """Hit Ratio metric.

    This class computes the hit ratio of a price prediction task.
    It measures the proportion of times the predicted price direction matches the actual price direction.
    """

    def __init__(self, **kwargs):
        super(HitRatio, self).__init__(**kwargs)

    def __call__(self,
                 y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor]
                 ) -> Union[float, torch.Tensor]:
        """Compute the hit ratio from predictions and targets."""
        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            # Remove nan/inf values (must do before diff, so both seq length match)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            # Not enough samples
            if len(y_true) < 2 or len(y_pred) < 2:
                return 0.0

            # Calculate actual and predicted price changes (difference with previous)
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)

            # Calculate sign (1: up, -1: down, 0: flat)
            true_sign = np.sign(true_diff)
            pred_sign = np.sign(pred_diff)

            # Calculate hit (direction correct)
            hits = (true_sign == pred_sign)
            hit_ratio_score = np.mean(hits)

            hit_ratio_score = hit_ratio_score * 100  # Convert to percentage

            return hit_ratio_score
        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            # Not enough samples
            if len(y_true) < 2 or len(y_pred) < 2:
                return torch.tensor(0.0, device=y_true.device)

            # Calculate actual and predicted price changes (difference with previous)
            true_diff = torch.diff(y_true)
            pred_diff = torch.diff(y_pred)
            # Calculate sign (1: up, -1: down, 0: flat)
            true_sign = torch.sign(true_diff)
            pred_sign = torch.sign(pred_diff)
            # Calculate hit (direction correct)
            hits = (true_sign == pred_sign)

            hit_ratio_score = torch.mean(hits.float())
            hit_ratio_score = hit_ratio_score * 100

            return hit_ratio_score
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    hit_ratio_metric = HitRatio()

    # Example usage
    y_true = np.array([100, 102, 101, 105, 107])
    y_pred = np.array([100, 101, 103, 104, 108])
    hit_ratio = hit_ratio_metric(y_true, y_pred)
    print(f"Hit Ratio: {hit_ratio}")

    y_true_tensor = torch.tensor([100, 102, 101, 105, 107])
    y_pred_tensor = torch.tensor([100, 101, 103, 104, 108])
    hit_ratio_tensor = hit_ratio_metric(y_true_tensor, y_pred_tensor)
    print(f"Hit Ratio (Tensor): {hit_ratio_tensor.item()}")