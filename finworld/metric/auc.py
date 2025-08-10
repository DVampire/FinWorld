import torch
import numpy as np
from typing import Union
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import auroc

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class AUC(Metric):
    """
    AUC (Area Under the ROC Curve) metric.

    This class computes the AUC score for binary or multiclass classification tasks.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    Supports both probability and one-hot encoded predictions.
    """

    def __init__(self, average='macro', **kwargs):
        """
        Args:
            average (str): The averaging method for multi-class tasks. Usually 'macro'.
        """
        super(AUC, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[float, torch.Tensor]:
        """
        Compute the AUC score from predictions and targets.
        Args:
            y_true: True labels, can be a numpy array or a torch tensor.
            y_pred: The predicted labels, can be a numpy array or a torch tensor.
        Returns:
            float or torch.Tensor: The accuracy score.
        """
        # Convert tensors to numpy arrays
        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):

            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)

            if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
                # Binary classification
                auc = roc_auc_score(y_true, y_pred)
            else:
                # Multiclass classification
                auc = roc_auc_score(
                    y_true,
                    y_pred,
                    average=self.average,
                    multi_class="ovr"  # One-vs-Rest strategy
                )

            auc = auc * 100  # Convert to percentage

            return float(auc)

        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            num_classes = y_pred.shape[-1] if y_pred.ndim > 1 \
                else max(torch.max(y_pred).item(), torch.max(y_true).item()) + 1

            if y_pred.ndim == 1:
                auc = auroc(y_pred, y_true.long(), task="binary", num_classes=2)
            else:
                auc = auroc(
                    y_pred,
                    y_true.long(),
                    task="multiclass",
                    num_classes=num_classes,
                    average=self.average
                )

            auc = auc * 100

            return auc

if __name__ == '__main__':
    auc_metric = AUC()

    # Example usage
    y_true = np.array([0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0.9, 0.8, 0.1, 0.7, 0])
    auc = auc_metric(y_true, y_pred)
    print(f"AUC (Numpy): {auc:.2f}%")

    y_true_tensor = torch.tensor([0, 1, 0, 0, 1, 0])
    y_pred_tensor = torch.tensor([0, 0.9, 0.8, 0.1, 0.7, 0])
    auc_tensor = auc_metric(y_true_tensor, y_pred_tensor)
    print(f"AUC (Tensor): {auc_tensor:.2f}%")

    y_true_tensor_multiclass = torch.tensor([0, 1, 2, 1, 0, 1])
    y_pred_tensor_multiclass = torch.tensor([[0, 0.9, 0.1],
                                             [0.8, 0.1, 0.1],
                                             [0.2, 0.7, 0.1],
                                             [0.1, 0.8, 0.1],
                                             [0.9, 0.1, 0],
                                             [0.2, 0.6, 0.2]])
    auc_multiclass = auc_metric(y_true_tensor_multiclass, y_pred_tensor_multiclass)
    print(f"Multiclass AUC (Tensor): {auc_multiclass:.2f}%")

    y_true_nf_multiclass = torch.tensor([0, 1, 2])
    y_pred_nf_multiclass = torch.from_numpy(np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]]))
    auc_nf_multiclass = auc_metric(y_true_nf_multiclass, y_pred_nf_multiclass)
    print(f"Multiclass AUC NF (Tensor): {auc_nf_multiclass:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    auc_nf = auc_metric(y_true_nf, y_pred_nf)
    print(f"Multiclass AUC NF (Numpy): {auc_nf:.2f}%")
