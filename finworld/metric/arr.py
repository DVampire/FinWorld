import torch
import numpy as np
from typing import List, Optional, Any, Union, Dict

from finworld.registry import METRIC
from finworld.calendar import calendar_manager
from finworld.metric.utils import clean_invalid_values
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class ARR(Metric):
    """Annualized Return Rate (ARR) metric.

    This class computes the Annualized Return Rate based on the returns of a financial asset.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 **kwargs
                 ):
        """
        Initialize the ARR metric.

        Args:
            level (str): The time level for which the ARR is computed. Default is "1day".
        """
        super(ARR, self).__init__(**kwargs)
        self.level = level
        self._symbol_info = symbol_info

    def __call__(self,
                 ret: Union[np.ndarray, torch.Tensor]
                 ) -> Union[float, torch.Tensor]:
        """Compute the ARR from returns.

        Args:
            ret (Union[np.ndarray, torch.Tensor]): Returns of the asset.
        Returns:
            float: The computed Annualized Return Rate.
        """
        if isinstance(ret, np.ndarray):
            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level = self.level)

            arr = (np.prod(1 + ret) ** (num_periods / len(ret))) - 1
            arr = arr * 100  # Convert to percentage

            return float(arr)
        elif isinstance(ret, torch.Tensor):
            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level = self.level)

            arr = (torch.prod(1 + ret) ** (num_periods / len(ret))) - 1
            arr = arr * 100
            return arr
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    method = ARR(
        level="1day",
        symbol_info={
            "symbol": "AAPL",
            "exchange": "New York Stock Exchange",
        }
    )

    ret = np.array([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])

    res = method(ret)
    print("Result:", res)

    ret = torch.tensor([0.01, 0.02, -0.005, 0.03, np.nan, np.inf])
    res = method(ret)
    print("Result (Tensor):", res)