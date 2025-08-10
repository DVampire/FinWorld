import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Any, Union, Dict

from finworld.registry import METRIC
from finworld.calendar import calendar_manager
from finworld.metric.utils import clean_invalid_values
from finworld.metric.dd import DD
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class SOR(Metric):
    """Sharpe Ratio (SOR) metric.

    This class computes the Sharpe Ratio based on the returns of a financial asset.
    It handles NaN and infinite values by removing them from the returns array.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: Dict[str, Any] = None,
                 risk_free_rate: float = 0.0,
                 **kwargs
                 ):
        """
        Initialize the SOR metric.

        Args:
            level (str): The time level for which the SOR is computed. Default is "1day".
            risk_free_rate (float): The risk-free rate to use in the Sharpe Ratio calculation. Default is 0.0.
        """
        super(SOR, self).__init__(**kwargs)

        self.level = level
        self._symbol_info = symbol_info
        self.risk_free_rate = risk_free_rate

        self.dd = DD(level=level, symbol_info=symbol_info)


    def __call__(self,
                 ret: Union[np.ndarray, torch.Tensor]
                 ) -> Union[float, torch.Tensor]:
        """Compute the SOR from returns.

        Args:
            ret (Union[np.ndarray, torch.Tensor]): Returns of the asset.
        Returns:
            float: The computed Sharpe Ratio.
        """
        if isinstance(ret, np.ndarray):

            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level=self.level)

            rf_period = self.risk_free_rate / num_periods
            excess_mean_period = ret.mean() - rf_period
            mu_ann = excess_mean_period * num_periods

            sigma_down_ann = self.dd(ret)

            sor = mu_ann / (sigma_down_ann + 1e-12)

            return float(sor)
        elif isinstance(ret, torch.Tensor):

            # process nan and inf
            ret = clean_invalid_values(ret)

            num_periods = calendar_manager.get_num_periods(symbol_info=self._symbol_info,
                                                           level=self.level)

            rf_period = self.risk_free_rate / num_periods
            excess_mean_period = ret.mean() - rf_period
            mu_ann = excess_mean_period * num_periods

            sigma_down_ann = self.dd(ret)

            sor = mu_ann / (sigma_down_ann + 1e-12)

            return sor

        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

if __name__ == '__main__':
    method = SOR(
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