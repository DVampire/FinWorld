import torch
import numpy as np
from typing import Union
from sklearn.metrics import accuracy_score
from torchmetrics.functional import accuracy

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class Accuracy(Metric):
    """
    Accuracy metric.

    This class computes the accuracy of a classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both binary and multiclass predictions.
    """

    def __init__(self, **kwargs):
        super(Accuracy, self).__init__(**kwargs)

    def __call__(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[float, torch.Tensor]:
        """
        Calculate the accuracy of predictions against true labels.
        Args:
            y_true: True labels, can be a numpy array or a torch tensor.
            y_pred: The predicted labels, can be a numpy array or a torch tensor.
        Returns:
            float or torch.Tensor: The accuracy score.
        """

        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=-1)

            # Compute accuracy
            acc = accuracy_score(y_true, y_pred)

            acc = acc * 100  # Convert to percentage

            return float(acc)

        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            num_classes = y_pred.shape[-1] if y_pred.ndim > 1 \
                else max(torch.max(y_pred).item(), torch.max(y_true).item()) + 1

            # Compute accuracy
            acc = accuracy(y_pred,
                           y_true,
                           task="multiclass",
                           num_classes=num_classes)

            acc = acc * 100  # Convert to percentage

            return acc

        else:
            raise TypeError("y_true and y_pred must be either numpy arrays or torch tensors.")

if __name__ == '__main__':
    accuracy_metric = Accuracy()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    acc = accuracy_metric(y_true, y_pred)
    print(f"Accuracy (Numpy): {acc:.2f}%")

    y_true_tensor = torch.tensor([0, 1, 2, 2, 1, 0])
    y_pred_tensor = torch.tensor([0, 1, 2, 1, 1, 0])
    acc_tensor = accuracy_metric(y_true_tensor, y_pred_tensor)
    print(f"Accuracy (Tensor): {acc_tensor:.2f}%")

    y_true_tensor_multiclass = torch.tensor([0, 1, 2])
    y_pred_tensor_multiclass = torch.tensor([[0, 0.8, 0], [0.1, 0, 0], [0, 0.2, 0.8]])
    acc_multiclass = accuracy_metric(y_true_tensor_multiclass, y_pred_tensor_multiclass)
    print(f"Multiclass Accuracy (Tensor): {acc_multiclass:.2f}%")

    y_true_nf_multiclass = torch.tensor([0, 1, 2])
    y_pred_nf_multiclass = torch.from_numpy(np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]]))
    acc_nf_multiclass = accuracy_metric(y_true_nf_multiclass, y_pred_nf_multiclass)
    print(f"Multiclass Accuracy NF (Tensor): {acc_nf_multiclass:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    acc_nf = accuracy_metric(y_true_nf, y_pred_nf)
    print(f"Multiclass Accuracy NF (Numpy): {acc_nf:.2f}%")
