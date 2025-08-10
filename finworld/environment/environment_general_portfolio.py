import warnings

from pandas import DataFrame

warnings.filterwarnings("ignore")

import numpy as np
from typing import Any, Dict, List
import random
import gym
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

from finworld.registry import ENVIRONMENT
from finworld.registry import DATASET
from finworld.utils import PortfolioRecords
from finworld.environment.wrapper import make_env
from finworld.utils import get_start_end_timestamp
from finworld.utils import TimeLevel, TimeLevelFormat

@ENVIRONMENT.register_module(force=True)
class EnvironmentGeneralPortfolio(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 history_timestamps: int = 32,
                 step_timestamps: int = 1,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 gamma: float = 0.99,
                 **kwargs
                 ):
        super(EnvironmentGeneralPortfolio, self).__init__()
        
        self.mode = mode
        self.dataset = dataset
        self.symbols = self.dataset.symbols
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.start_timestamp, self.end_timestamp = get_start_end_timestamp(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level
        )

        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.gamma = gamma
        self.action_dim = 1 + len(self.symbols)  # cash + assets

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.original_prices_dfs = self.res_info['original_prices_dfs']

    def _init_features(self):

        timestamp_info = {}
        assets_meta_info = self.dataset.assets_meta_info['items']
        for key, value in assets_meta_info.items():
            start_timestamp = value["history_info"]["start_timestamp"]
            end_timestamp = value["history_info"]["end_timestamp"]

            if (end_timestamp >= self.start_timestamp
                    and end_timestamp <= self.end_timestamp):
                timestamp_info[key] = {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }

        self.timestamp_min_index = min(timestamp_info.keys())
        self.timestamp_max_index = max(timestamp_info.keys())
        self.timestamp_min = timestamp_info[self.timestamp_min_index]["start_timestamp"]
        self.timestamp_max = timestamp_info[self.timestamp_max_index]["end_timestamp"]

        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1
        assert self.num_timestamps == len(
            timestamp_info), f"num_timestamps {self.num_timestamps} != len(data_info) {len(timestamp_info)}"

        features_dfs = {}
        prices_dfs = {}
        times_dfs = {}
        original_prices_dfs = {}
        labels_dfs = {}
        news_dfs = {}
        for symbol in self.symbols:
            features_dfs[symbol] = self.dataset.assets_data[symbol]["features"]
            prices_dfs[symbol] = self.dataset.assets_data[symbol]["prices"]
            times_dfs[symbol] = self.dataset.assets_data[symbol]["times"]
            original_prices_dfs[symbol] = self.dataset.assets_data[symbol]["original_prices"]
            labels_dfs[symbol] = self.dataset.assets_data[symbol]["labels"]
            news_dfs[symbol] = self.dataset.assets_data[symbol]["news"]

        res_info = dict(
            timestamp_info=timestamp_info,
            features_dfs=features_dfs,
            prices_dfs=prices_dfs,
            original_prices_dfs=original_prices_dfs,
            times_dfs=times_dfs,
            labels_dfs=labels_dfs,
            news_dfs=news_dfs,
        )

        return res_info
    
    def _get_dataitem(self,
                      df: DataFrame,
                      start_timestamp: str,
                      end_timestamp: str):
        df = deepcopy(df)
        df = df[(start_timestamp <= df.index) & (df.index <= end_timestamp)]
        return df

    def _init_timestamp_index(self):
        if self.mode == "train":
            timestamp_index = random.randint(self.timestamp_min_index,
                                             self.timestamp_min_index + 3 * (self.num_timestamps // 4))
        else:
            timestamp_index = self.timestamp_min_index
        return timestamp_index

    def get_timestamp_string(self, timestamp_index: int):
        end_timestamp = self.timestamp_info[timestamp_index]["end_timestamp"]
        end_timestamp_string = end_timestamp.strftime(self.level_format.value)
        return end_timestamp_string
    
    def get_value(self,
                  cash,
                  position,
                  price
                  ):
        value = cash + np.sum(position * price)
        return value
    
    def get_price(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        
        prices = []
        for symbol in self.symbols:
            prices_df = self.original_prices_dfs[symbol]
            prices_df = self._get_dataitem(prices_df, start_timestamp, end_timestamp)
            
            prices_dict = prices_df.iloc[-1].to_dict()

            # close, high, low, open, volume
            close, high, low, open, volume = (prices_dict["close"],
                                              prices_dict["high"],
                                              prices_dict["low"],
                                              prices_dict["open"],
                                              prices_dict["volume"])
            price = close
            
            prices.append(price)

        return np.array(prices)

    def get_data(self):
        start_timestamp_index = self.timestamp_min_index
        end_timestamp_index = self.timestamp_max_index

        if self.level == TimeLevel.DAY:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(days=1)
        elif self.level == TimeLevel.HOUR:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(hours=1)
        elif self.level == TimeLevel.MINUTE:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(minutes=1)
        else:
            start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"] - pd.Timedelta(seconds=1)

        end_timestamp = self.timestamp_info[end_timestamp_index]["end_timestamp"]

        features_dfs = {}
        prices_dfs = {}
        original_prices_dfs = {}
        times_dfs = {}
        labels_dfs = {}
        news_dfs = {}

        for symbol in self.symbols:

            prices_df = self._get_dataitem(
                self.res_info['prices_dfs'][symbol],
                start_timestamp,
                end_timestamp
            )
            prices_dfs[symbol] = prices_df

            original_prices_df = self._get_dataitem(
                self.res_info['original_prices_dfs'][symbol],
                start_timestamp,
                end_timestamp
            )
            original_prices_dfs[symbol] = original_prices_df

            labels_df = self._get_dataitem(
                self.res_info['labels_dfs'][symbol],
                start_timestamp,
                end_timestamp
            )
            labels_dfs[symbol] = labels_df

            if self.res_info['features_dfs'][symbol] is not None:
                features_df = self._get_dataitem(
                    self.res_info['features_dfs'][symbol],
                    start_timestamp,
                    end_timestamp
                )
            else:
                features_df = None
            features_dfs[symbol] = features_df

            if self.res_info['times_dfs'][symbol] is not None:
                times_df = self._get_dataitem(
                    self.res_info['times_dfs'][symbol],
                    start_timestamp,
                    end_timestamp
                )
            else:
                times_df = None
            times_dfs[symbol] = times_df

            if self.res_info['news_dfs'][symbol] is not None:
                news_df = self._get_dataitem(
                    self.res_info['news_dfs'][symbol],
                    start_timestamp,
                    end_timestamp
                )
            else:
                news_df = None
            news_dfs[symbol] = news_df

        data = dict(
            features_dfs=features_dfs,
            prices_dfs=prices_dfs,
            original_prices_dfs=original_prices_dfs,
            times_dfs=times_dfs,
            labels_dfs=labels_dfs,
            news_dfs=news_dfs,
        )

        return data
    
    def execute_action(self, 
                       cash,
                       position,
                       price,
                       action):
        
        cash_ratio = action[0]
        assets_ratios = action[1:]
        
        value = self.get_value(cash = cash,
                               position = position,
                               price = price)
        
        cash = cash_ratio * value
        assets_values = assets_ratios * value
        position = assets_values / price

        value = cash + np.sum(assets_values)

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
        }
        
        return res_info

    def get_state(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        prices = {}
        for symbol in self.symbols:
            prices_df = self.original_prices_dfs[symbol]
            prices_df = self._get_dataitem(prices_df, start_timestamp, end_timestamp)
            prices[symbol] = prices_df

        state = dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            prices = prices,
        )

        return state

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)
        
        self.ret = 0
        self.cash = self.initial_amount
        self.position = np.zeros(len(self.symbols))
        self.discount = 1.0
        self.value = self.initial_amount
        self.pre_value = self.value
        self.total_return = 0
        self.total_profit = 0
        self.action = np.array([1.0] + [0.0] * len(self.symbols))
        self.done = False

        self.state = self.get_state(timestamp_index=self.timestamp_index)
        
        info = {
            "timestamp": self.timestamp_string,
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "pre_value": self.pre_value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "done": self.done,
        }
        
        return self.state, info
    
    def step(self, action):
        
        action = np.array(action)
        
        res_info = self.execute_action(cash = self.cash,
                                       position = self.position,
                                       price = self.price,
                                       action = action)
        
        self.cash = res_info["cash"]
        self.position = res_info["position"]
        self.value = res_info["value"]
        self.action = res_info["action"]
        
        ret = (self.value - self.pre_value) / self.pre_value if self.pre_value != 0 else 0
        
        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100
        
        # next timestamp
        self.timestamp_index += 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncated = False
        else:
            self.done = True
            self.truncated = True
            
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        self.state = self.get_state(timestamp_index=self.timestamp_index)
        
        reward = ret
        
        info = {
            "timestamp": self.timestamp_string,
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "value": self.value,
            "pre_value": self.pre_value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "done": self.done,
        }
        
        self.pre_value = self.value
        
        return self.state, reward, self.done, self.truncated, info
    
if __name__ == "__main__":
    num_envs = 2
    history_timestamps = 64
    future_timestamps = 32
    start_timestamp = "2023-05-01"
    end_timestamp = "2025-05-01"
    action_dim = 1 + 29
    num_features = 150
    num_times = 4
    num_assets = 29
    num_block_tokens = history_timestamps
    
    dataset = dict(
        type="MultiAssetDataset",
        assets_name="dj30",
        data_path="datasets/dj30",
        enabled_data_configs=[
            {
                "asset_name": "dj30",
                "source": "fmp",
                "data_type": "price",
                "level": "1day",
            },
            {
                "asset_name": "dj30",
                "source": "fmp",
                "data_type": "feature",
                "level": "1day",
            },
            {
                "asset_name": "dj30",
                "source": "fmp",
                "data_type": "news",
                "level": "1day",
            },
            {
                "asset_name": "dj30",
                "source": "alpaca",
                "data_type": "news",
                "level": "1day",
            }
        ],
        if_norm=True,
        if_use_temporal=True,
        if_norm_temporal=True,
        if_use_features=True,
        if_use_rank=True,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        history_timestamps=64,
        future_timestamps=32,
        start_timestamp="2015-05-01",
        end_timestamp= "2025-05-01",
        level="1day",
    )
    
    environment = dict(
        type="EnvironmentGeneralPortfolio",
        mode="test",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
    )

    dataset = DATASET.build(dataset)

    environment.update({
        "dataset": dataset
    })

    environment = ENVIRONMENT.build(environment)

    records = PortfolioRecords()

    state, info = environment.reset()

    records.add(
        dict(
            timestamp=info["timestamp"],
            price=info["price"],
            cash=info["cash"],
            position=info["position"],
            value=info["value"],
        ),
    )

    while True:
        action = np.abs(np.random.randn(action_dim).astype(np.float32))
        action = action / np.sum(action)

        next_state, reward, done, truncted, info = environment.step(action)

        records.add(
            dict(
                action=info["action"],
                ret=info["ret"],
                total_profit=info["total_profit"],
                timestamp=info["timestamp"],  # next timestamp
                price=info["price"],  # next price
                cash=info["cash"],  # next cash
                position=info["position"],  # next position
                value=info["value"],  # next value
            ),
        )

        if done or truncted:
            break

    records.add(
        dict(
            action=info["action"],
            ret=info["ret"],
            total_profit=info["total_profit"],
        )
    )

    print(records.to_dataframe())