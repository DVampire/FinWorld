import torch
import numpy as np
from typing import Any, Union

from finworld.registry import METRIC
from finworld.metric.utils import clean_invalid_values
from finworld.calendar import calendar_manager
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class DD(Metric):
    """Downside Deviation (DD) metric.
    This class computes the Downside Deviation based on the returns of a financial asset.
    It handles NaN and infinite values by replacing them with zero.
    """
    def __init__(self,
                    level: str = "1day",
                    symbol_info: Any = None,
                    **kwargs
                 ):
            """
            Initialize the DD metric.

            Args:
                level (str): The time level for which the DD is computed. Default is "1day".
                symbol_info (Any): Information about the financial asset.
            """

            super(DD, self).__init__(**kwargs)

            self.level = level
            self._symbol_info = symbol_info

    def __call__(self,
                 ret: Union[np.ndarray, torch.Tensor],
                 target: float = 0.0
                 ) -> Union[float, torch.Tensor]:
        """Compute the Downside Deviation from returns.

        Args:
            ret (Union[np.ndarray, torch.Tensor]): Returns of the asset.
            target (float): The target return to compare against. Default is 0.0.

        Returns:
            float: The computed Downside Deviation.
        """
        if isinstance(ret, np.ndarray):

            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level = self.level)

            # Calculate the downside returns
            downside_returns = np.where(ret < target, ret - target, 0.0)
            # Calculate the downside deviation
            dd = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(num_periods)

            return float(dd)
        elif isinstance(ret, torch.Tensor):

            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level = self.level)

            # Calculate the downside returns
            downside_returns = torch.where(ret < target, ret - target, torch.tensor(0.0, device=ret.device))
            # Calculate the downside deviation
            dd = torch.sqrt(torch.mean(downside_returns ** 2)) * torch.sqrt(torch.tensor(num_periods, device=ret.device))

            return dd

        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    method = DD(
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