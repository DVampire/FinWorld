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

@AGENT.register_module(force=True)
class BuyAndHold():
    def __init__(self,
                 task_type: str,
                 level: str = "1day",
                 **kwargs
                 ):
        super(BuyAndHold, self).__init__()

        self.task_type = TaskType.from_string(task_type)
        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)

    def forward(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Buy and hold strategy: always buy the stock at the first time step and hold it.

        Args:
            data (pd.DataFrame): The input data containing stock prices and other features.
                The DataFrame should have a column for stock prices, typically named "price".

        Returns:
            List[str]: A list of actions, where each action is "buy" for the first stock.
        """

        if self.task_type == TaskType.TRADING:
            data = data.copy()
            data[['close', 'open', 'high', 'low', 'volume']] = data[['close', 'open', 'high', 'low', 'volume']].astype(
                float)

            res = {}

            data = data.reset_index(drop=False)
            timestamps = data['timestamp'].values
            num_timesteps = len(timestamps)

            actions = [2] + [1] * (num_timesteps - 1) # BUY at the first timestep, the HOLD for the rest
            actions = np.array(actions, dtype=np.int32)

            for timestamp, action in zip(timestamps, actions):
                timestamp = pd.to_datetime(timestamp)
                timestamp = timestamp.strftime(self.level_format.value)
                res[timestamp] = action

            return res

        elif self.task_type == TaskType.PORTFOLIO:
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

            actions = []
            for i in range(num_timesteps):
                action = np.array([.0] + [1.0 / num_assets] * num_assets) # Average allocation to each asset
                actions.append(action)
            actions = np.array(actions, dtype=np.float32)

            for timestamp, action in zip(timestamps, actions):
                timestamp = pd.to_datetime(timestamp)
                timestamp = timestamp.strftime(self.level_format.value)
                res[timestamp] = action

            return res