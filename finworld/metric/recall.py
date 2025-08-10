import torch
import numpy as np
from typing import Union
from sklearn.metrics import recall_score
from torchmetrics.functional import recall

from finworld.registry import METRIC
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class Recall(Metric):
    """Recall metric.

    This class computes the recall of a classification task.
    It handles NaN and infinite values by removing them from the predictions and targets arrays.
    """

    def __init__(self,
                 average='macro',
                 **kwargs):
        super(Recall, self).__init__(**kwargs)
        self.average = average

    def __call__(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[float, torch.Tensor]:
        """
        Calculate the recall of predictions against true labels.
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
            rec = recall_score(y_true, y_pred, average=self.average)

            rec = rec * 100  # Convert to percentage

            return float(rec)

        elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):

            num_classes = y_pred.shape[-1] if y_pred.ndim > 1 \
                else max(torch.max(y_pred).item(), torch.max(y_true).item()) + 1

            # Compute recall
            rec = recall(y_pred,
                         y_true,
                         task="multiclass",
                         num_classes=num_classes,
                         average=self.average)

            rec = rec * 100  # Convert to percentage

            return rec

if __name__ == '__main__':
    recall_metric = Recall()

    # Example usage
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    rec = recall_metric(y_true, y_pred)
    print(f"Recall (Numpy): {rec:.2f}%")

    y_true_tensor = torch.tensor([0, 1, 2, 2, 1, 0])
    y_pred_tensor = torch.tensor([0, 1, 2, 1, 1, 0])
    rec_tensor = recall_metric(y_true_tensor, y_pred_tensor)
    print(f"Recall (Tensor): {rec_tensor:.2f}%")

    y_true_tensor_multiclass = torch.tensor([0, 1, 2])
    y_pred_tensor_multiclass = torch.tensor([[0, 0.8, 0], [0.1, 0, 0], [0, 0.2, 0.8]])
    rec_multiclass = recall_metric(y_true_tensor_multiclass, y_pred_tensor_multiclass)
    print(f"Multiclass Recall (Tensor): {rec_multiclass:.2f}%")

    y_true_nf_multiclass = torch.tensor([0, 1, 2])
    y_pred_nf_multiclass = torch.from_numpy(np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]]))
    rec_nf_multiclass = recall_metric(y_true_nf_multiclass, y_pred_nf_multiclass)
    print(f"Multiclass Recall NF (Tensor): {rec_nf_multiclass:.2f}%")

    y_true_nf = np.array([0, 1, 2])
    y_pred_nf = np.array([[0, 1, 0], [np.nan, 1, 0], [0, 0, np.inf]])
    rec_nf = recall_metric(y_true_nf, y_pred_nf)
    print(f"Multiclass Recall NF (Numpy): {rec_nf:.2f}%")

