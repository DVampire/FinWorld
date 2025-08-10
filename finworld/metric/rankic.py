import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Any, Union

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class RANKIC(Metric):
    """Rank Information Coefficient (RankIC) metric.

    This class computes the Rank Information Coefficient between true and predicted values.
    It handles NaN and infinite values by replacing them with zero.
    """
    def __init__(self, **kwargs):
        """
        Initialize the RankIC metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(RANKIC, self).__init__(**kwargs)

    def __call__(self,
                 y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor],
                 mask: Optional[Union[np.ndarray, torch.Tensor]] = None
                 ) -> Union[float, torch.Tensor]:
        """Compute the Rank Information Coefficient between true and predicted values.

        Args:
            y_true (Union[np.ndarray, torch.Tensor]): True values.
            y_pred (Union[np.ndarray, torch.Tensor]): Predicted values.
            mask (Optional[Union[np.ndarray, torch.Tensor]]): Optional mask to apply to the true and predicted values.
        Returns:
            float: The computed Rank Information Coefficient.
        """

        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()

            if mask is not None:
                mask = np.asarray(mask, dtype=bool).ravel()
                keep = ~mask
            else:
                keep = np.ones_like(y_true, dtype=bool)

            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            keep &= finite

            if keep.sum() < 2: # Not enough data to compute RankIC
                return .0

            y_true = y_true[keep]
            y_pred = y_pred[keep]

            true_rank = y_true.argsort().argsort().astype(float)
            pred_rank = y_pred.argsort().argsort().astype(float)

            cov = np.mean(
                (true_rank - true_rank.mean()) *
                (pred_rank - pred_rank.mean())
            )
            std = true_rank.std() * pred_rank.std()

            rank_ic = cov / (std + 1e-12)

            return float(rank_ic)

        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if mask is not None:
                mask = mask.view(-1).bool()
                keep = ~mask
            else:
                keep = torch.ones_like(y_true, dtype=torch.bool)

            finite = torch.isfinite(y_true) & torch.isfinite(y_pred)
            keep &= finite

            if keep.sum() < 2:
                return torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device)

            y_true = y_true[keep]
            y_pred = y_pred[keep]

            true_rank = y_true.argsort().argsort().float()
            pred_rank = y_pred.argsort().argsort().float()

            cov = torch.mean(
                (true_rank - true_rank.mean()) *
                (pred_rank - pred_rank.mean())
            )

            std_true = true_rank.std(unbiased=False)
            std_pred = pred_rank.std(unbiased=False)

            std = std_true * std_pred

            rank_ic = cov / (std + 1e-12)

            return rank_ic
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    rankic_metric = RANKIC()

    # Example usage with NumPy arrays
    y_true_np = np.array([1, 2, 3, 4, 5])
    y_pred_np = np.array([5, 4, 3, 2, 1])
    rankic_score_np = rankic_metric(y_true_np, y_pred_np)
    print(f"RankIC (Numpy): {rankic_score_np:.4f}")

    # Example usage with PyTorch tensors
    y_true_tensor = torch.tensor([1, 2, 3, 4, 5])
    y_pred_tensor = torch.tensor([5, 4, 3, 2, 1])
    rankic_score_tensor = rankic_metric(y_true_tensor, y_pred_tensor)
    print(f"RankIC (Tensor): {rankic_score_tensor:.4f}")