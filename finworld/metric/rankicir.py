import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Any, Union

from finworld.registry import METRIC
from finworld.metric.utils import fill_invalid_values
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class RANKICIR(Metric):
    """Rank Information Coefficient Information Ratio (RankICIR) metric.

    This class computes the Rank Information Coefficient between true and predicted values,
    and then calculates the Cumulative Information Ratio.
    It handles NaN and infinite values by replacing them with zero.
    """
    def __init__(self, **kwargs):
        """
        Initialize the RankICIR metric.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super(RANKICIR, self).__init__(**kwargs)

    def __call__(self,
                 rankic_values: Union[np.ndarray, torch.Tensor],
                 )-> Union[float, torch.Tensor]:
        """Compute the Rank Information Coefficient Information Ratio.
        Args:
            rankic_values (Union[np.ndarray, torch.Tensor]): RankIC values.
        Returns:
            float: The computed Cumulative Information Ratio.
        """

        if isinstance(rankic_values, np.ndarray):
            rankic_values = np.asarray(rankic_values).ravel()
            rankic_values = fill_invalid_values(rankic_values)

            randkicir = np.mean(rankic_values) / (np.std(rankic_values) + 1e-12)

            return float(randkicir)
        elif isinstance(rankic_values, torch.Tensor):
            rankic_values = rankic_values.view(-1)
            rankic_values = fill_invalid_values(rankic_values)

            randkicir = torch.mean(rankic_values) / (torch.std(rankic_values) + 1e-12)

            return randkicir
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    # Example usage
    rankicir_metric = RANKICIR()
    rankic_values = np.random.rand(100)  # Example RankIC values
    result = rankicir_metric(rankic_values)
    print(f"RankICIR: {result}")
