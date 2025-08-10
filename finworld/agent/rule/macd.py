import numpy as np
from typing import List, Union, Dict
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from finworld.registry import AGENT
from finworld.task import TaskType
from finworld.utils import TimeLevel, TimeLevelFormat


def macd_signal(close: pd.Series, fast=12, slow=26, signal=9):
    """
    Calculate MACD signals: 2=buy, 0=sell, 1=hold.
    Buy when MACD crosses above signal line (golden cross), sell when crosses below (death cross).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    actions = [1]  # Default: hold on the first timestep
    for i in range(1, len(close)):
        # Golden cross: MACD crosses above signal line
        if macd.iloc[i - 1] < signal_line.iloc[i - 1] and macd.iloc[i] > signal_line.iloc[i]:
            actions.append(2)  # Buy
        # Death cross: MACD crosses below signal line
        elif macd.iloc[i - 1] > signal_line.iloc[i - 1] and macd.iloc[i] < signal_line.iloc[i]:
            actions.append(0)  # Sell
        else:
            actions.append(1)  # Hold
    return actions


@AGENT.register_module(force=True)
class MACD():
    def __init__(self,
                 task_type: str,
                 level: str = "1day",
                 **kwargs
                 ):
        super(MACD, self).__init__()

        self.task_type = TaskType.from_string(task_type)
        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)

    def forward(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict:
        """
        MACD-based trading strategy: buy on golden cross, sell on death cross, hold otherwise.
        Returns a dictionary with timestamp as key and action as value.
        """
        if self.task_type == TaskType.TRADING:
            data = data.copy()
            # Ensure correct types
            data[['close', 'open', 'high', 'low', 'volume']] = data[['close', 'open', 'high', 'low', 'volume']].astype(
                float)

            res = {}

            data = data.reset_index(drop=False)
            timestamps = data['timestamp'].values
            close = data['close']

            actions = macd_signal(close)
            actions = np.array(actions, dtype=np.int32)

            for timestamp, action in zip(timestamps, actions):
                timestamp = pd.to_datetime(timestamp)
                timestamp = timestamp.strftime(self.level_format.value)
                res[timestamp] = action

            return res

        elif self.task_type == TaskType.PORTFOLIO:
            # For each asset, you can also calculate MACD and allocate by signal if desired.
            # This example still does equal allocation for demonstration.
            data = data.copy()
            data = {
                key: value[['close', 'open', 'high', 'low', 'volume']].astype(float) for key, value in data.items()
            }

            res = {}
            num_assets = len(data)
            first_asset = data[list(data.keys())[0]]
            first_asset = first_asset.reset_index(drop=False)
            timestamps = first_asset['timestamp'].values
            num_timesteps = len(timestamps)

            # (For demo) Allocate equally. You can modify this logic as needed.
            actions = []
            for i in range(num_timesteps):
                action = np.array([.0] + [1.0 / num_assets] * num_assets)
                actions.append(action)
            actions = np.array(actions, dtype=np.float32)

            for timestamp, action in zip(timestamps, actions):
                timestamp = pd.to_datetime(timestamp)
                timestamp = timestamp.strftime(self.level_format.value)
                res[timestamp] = action

            return res
