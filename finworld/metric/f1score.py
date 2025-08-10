import torch
import numpy as np
from typing import Union
from sklearn.metrics import f1_score
from torchmetrics.functional import f1_score as f1_torch

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class F1Score(Metric):
    """
    F1 Score metric.

    This class computes the F1 score for a classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both binary and multiclass predictions.
    """

    def __init__(self, average='macro', **kwargs):
        """
        Args:
            average (str): The averaging method for multi-class tasks.
                           Options: 'micro', 'macro', 'weighted', or None.
        """
        super(F1Score, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[float, torch.Tensor]:
        """
        Calculate the F1 of predictions against true labels.
        Args:
            y_true: True labels, can be a numpy array or a torch tensor.
            y_pred: The predicted labels, can be a numpy array or a torch tensor.
        Returns:
            float or torch.Tensor: The recall score.
        """

        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=-1)

            # Compute recall
            f1 = f1_score(y_true, y_pred, average=self.average, zero_division=.0)

            f1 = f1 * 100  # Convert to percentage

            return float(f1)

        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            num_classes = y_pred.shape[-1] if y_pred.ndim > 1 \
                else max(torch.max(y_pred).item(), torch.max(y_true).item()) + 1

            # Compute recall
            f1 = f1_torch(y_pred,
                          y_true,
                          task="multiclass",
                          num_classes=num_classes,
                          average=self.average,
                          zero_division=.0)

            f1 = f1 * 100  # Convert to percentage

            return f1

if __name__ == '__main__':
    f1_metric = F1Score()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    f1 = f1_metric(y_true, y_pred)
    print(f"F1 (Numpy): {f1:.2f}%")

    y_true_tensor = torch.tensor([0, 1, 2, 2, 1, 0])
    y_pred_tensor = torch.tensor([0, 1, 2, 1, 1, 0])
    f1_tensor = f1_metric(y_true_tensor, y_pred_tensor)
    print(f"F1 (Tensor): {f1_tensor:.2f}%")

    y_true_tensor_multiclass = torch.tensor([0, 1, 2])
    y_pred_tensor_multiclass = torch.tensor([[0, 0.8, 0], [0.1, 0, 0], [0, 0.2, 0.8]])
    f1_multiclass = f1_metric(y_true_tensor_multiclass, y_pred_tensor_multiclass)
    print(f"Multiclass F1 (Tensor): {f1_multiclass:.2f}%")

    y_true_nf_multiclass = torch.tensor([0, 1, 2])
    y_pred_nf_multiclass = torch.from_numpy(np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]]))
    f1_nf_multiclass = f1_metric(y_true_nf_multiclass, y_pred_nf_multiclass)
    print(f"Multiclass F1 NF (Tensor): {f1_nf_multiclass:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    f1_nf = f1_metric(y_true_nf, y_pred_nf)
    print(f"Multiclass F1 NF (Numpy): {f1_nf:.2f}%")
