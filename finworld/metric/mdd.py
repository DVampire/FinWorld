import torch
import numpy as np
from typing import Any, Union

from finworld.registry import METRIC
from finworld.metric.utils import clean_invalid_values
from finworld.metric.base import Metric

@METRIC.register_module(force=True)
class MDD(Metric):
    """Maximum Drawdown (MDD) metric.

    This class computes the Maximum Drawdown based on the returns of a financial asset.
    It handles NaN and infinite values by replacing them with zero.
    """

    def __init__(self,
                 level: str = "1day",
                 symbol_info: dict = None,
                 **kwargs
                 ):
        """
        Initialize the MDD metric.
        """
        super(MDD, self).__init__(**kwargs)
        self.level = level
        self._symbol_info = symbol_info


    def __call__(self,
                 ret: Union[np.ndarray, torch.Tensor]
                 ) -> Union[float, torch.Tensor]:
        """Compute the MDD from returns.

        Args:
            ret (Union[np.ndarray, torch.Tensor]): Returns of the asset.
        Returns:
            float: The computed Maximum Drawdown.
        """
        if isinstance(ret, np.ndarray):
            # process nan and inf
            ret = clean_invalid_values(ret)

            cumulative_returns = np.cumprod(1 + ret)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (peak + 1e-12)

            mdd = np.max(drawdown)

            mdd = mdd * 100

            return float(mdd)
        elif isinstance(ret, torch.Tensor):
            # process nan and inf
            ret = clean_invalid_values(ret)

            cumulative_returns = torch.cumprod(1 + ret, dim=0)
            peak, _ = torch.cummax(cumulative_returns, dim=0)
            drawdown = (peak - cumulative_returns) / (peak + 1e-12)

            mdd = torch.max(drawdown)

            mdd = mdd * 100

            return mdd

        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


if __name__ == '__main__':
    method = MDD(
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