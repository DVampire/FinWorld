import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict

from pandas import DataFrame

warnings.filterwarnings('ignore')

from typing import Any, Optional
from mathruler.grader import extract_boxed_content
import random

import gym
import numpy as np
import pandas as pd

from finworld.registry import DATASET, ENVIRONMENT
from finworld.utils import TradingRecords
from finworld.utils import get_start_end_timestamp
from finworld.utils import TimeLevel


__all__ = ['EnvironmentGeneralTrading']


@ENVIRONMENT.register_module(force=True)
class EnvironmentGeneralTrading(gym.Env):
    def __init__(
        self,
        *args,
        mode: str = "train",
        dataset: Any = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        future_timestamps: int = 32,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        gamma: float = 0.99,
        **kwargs,
    ):
        super(EnvironmentGeneralTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.symbol = self.dataset.symbol
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

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_df = self.res_info['features_df']
        self.original_prices_df = self.res_info['original_prices_df']
        self.labels_df = self.res_info['labels_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

    def _init_features(self):

        timestamp_info = {}
        asset_meta_info = self.dataset.asset_meta_info['items']
        for key, value in asset_meta_info.items():
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

        features_df = self.dataset.asset_data["features"]
        prices_df = self.dataset.asset_data["prices"]
        times_df = self.dataset.asset_data["times"]
        original_prices_df = self.dataset.asset_data["original_prices"]
        labels_df = self.dataset.asset_data["labels"]
        news_df = self.dataset.asset_data["news"]

        res_info = dict(
            timestamp_info=timestamp_info,
            features_df=features_df,
            prices_df=prices_df,
            original_prices_df=original_prices_df,
            times_df=times_df,
            labels_df=labels_df,
            news_df=news_df,
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
                  cash: float,
                  postition: int,
                  price: float):
        value = cash + postition * price
        return value

    def get_price(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        original_prices_df = self._get_dataitem(self.original_prices_df,
                                       start_timestamp,
                                       end_timestamp)

        prices = original_prices_df.iloc[-1].to_dict()

        # close, high, low, open, volume
        close, high, low, open, volume = (prices["close"],
                                          prices["high"],
                                          prices["low"],
                                          prices["open"],
                                          prices["volume"])
        price = close

        return price

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


        prices_df = self._get_dataitem(self.res_info['prices_df'],
                                        start_timestamp,
                                        end_timestamp)
        original_prices_df = self._get_dataitem(self.res_info['original_prices_df'],
                                                start_timestamp,
                                                end_timestamp)
        labels_df = self._get_dataitem(self.res_info['labels_df'],
                                        start_timestamp,
                                        end_timestamp)
        if self.res_info['features_df'] is not None:
            features_df = self._get_dataitem(self.res_info['features_df'],
                                         start_timestamp,
                                         end_timestamp)
        else:
            features_df = None
        if self.res_info['times_df'] is not None:
            times_df = self._get_dataitem(self.res_info['times_df'],
                                          start_timestamp,
                                          end_timestamp)
        else:
            times_df = None
        if self.res_info['news_df'] is not None:
            news_df = self._get_dataitem(self.res_info['news_df'],
                                         start_timestamp,
                                         end_timestamp)
        else:
            news_df = None

        data = dict(
            features_df=features_df,
            original_prices_df=original_prices_df,
            prices_df=prices_df,
            labels_df=labels_df,
            times_df=times_df,
            news_df=news_df,
        )

        return data

    def get_price_full(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        original_prices_df = self._get_dataitem(self.original_prices_df,
                                       start_timestamp,
                                       end_timestamp)

        prices = original_prices_df.iloc[-1].to_dict()

        # close, high, low, open, volume
        close, high, low, open, volume = (prices["close"],
                                          prices["high"],
                                          prices["low"],
                                          prices["open"],
                                          prices["volume"])

        return close, high, low, open, volume

    def get_state(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)

        state = dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            price = price
        )

        return state

    def eval_buy_position(self,
                          cash: float,
                          price: float):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self,
                           position: int):
        # evaluate sell position
        return int(position)

    def buy(self,
            cash: float,
            position: int,
            price: float,
            amount: int):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price=price, cash=cash)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount)) * eval_buy_postion))

        cash = cash - (buy_position * price * (1 + self.transaction_cost_pct))
        position = position + buy_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if buy_position == 0:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "BUY"
            action = self.action_labels.index("BUY")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def sell(self,
             cash: float,
             position: int,
             price: float,
             amount: int):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position(position=position)

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount)) * eval_sell_postion))

        cash = cash + (sell_position * price * (1 - self.transaction_cost_pct))
        position = position - sell_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if sell_position == 0:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "SELL"
            action = self.action_labels.index("SELL")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def hold(self,
             cash: float,
             position: int,
             price: float,
             amount: int):

        value = self.get_value(cash=cash, postition=position, price=price)

        action_label = "HOLD"
        action = self.action_labels.index("HOLD")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        self.ret = 0.0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.action = 1
        self.action_label = 'HOLD'
        self.done = False

        # after init record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        return self.state, info

    def _extract_action(self, action: str):
        extract = extract_boxed_content(action)
        if extract in self.action_labels:
            action = extract
        else:
            action = 'HOLD'
        action = self.action_labels.index(action)
        return action

    def step(self, action: Any):

        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, str):
            action = self._extract_action(action)
        elif isinstance(action, int):
            action = int(action)

        action = action - 1  # modify the action to -1, 0, 1

        if action > 0:
            res_info = self.buy(cash=self.cash,
                                position=self.position,
                                price=self.price,
                                amount=action)
        elif action < 0:
            res_info = self.sell(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=action)
        else:
            res_info = self.hold(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=action)

        self.cash = res_info['cash']
        self.position = res_info['position']
        self.value = res_info['value']
        self.action = res_info['action']
        self.action_label = res_info['action_label']

        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)

        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        # next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        # after update record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        reward = ret

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )

        # update the pre_value
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncted, info


if __name__ == '__main__':
    symbol = "AAPL"
    history_timestamps = 64
    future_timestamps = 32
    start_timestamp = "2023-05-01"
    end_timestamp = "2025-05-01"

    dataset = dict(
        type="SingleAssetDataset",
        symbol=symbol,
        data_path="datasets/exp",
        enabled_data_configs=[
            {
                "asset_name": "exp",
                "source": "fmp",
                "data_type": "price",
                "level": "1day",
            },
            {
                "asset_name": "exp",
                "source": "fmp",
                "data_type": "feature",
                "level": "1day",
            },
            {
                "asset_name": "exp",
                "source": "fmp",
                "data_type": "news",
                "level": "1day",
            },
            {
                "asset_name": "exp",
                "source": "alpaca",
                "data_type": "news",
                "level": "1day",
            }
        ],
        if_norm=True,
        if_use_future=True,
        if_use_temporal=True,
        if_norm_temporal=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )

    env_cfg: dict[str, Any] = dict(
        type='EnvironmentGeneralTrading',
        mode = "test",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        history_timestamps=history_timestamps,
        step_timestamps=1,
        future_timestamps=future_timestamps,
        start_timestamp="2023-05-01",
        end_timestamp="2025-05-01",
        gamma=0.99,
    )

    dataset = DATASET.build(dataset)

    env_cfg.update(
        dict(
            dataset=dataset,
        )
    )

    record = TradingRecords()

    environment = ENVIRONMENT.build(env_cfg)

    state, info = environment.reset()

    record.add(
        dict(
            timestamp=info["timestamp"],
            price=info["price"],
            cash=info["cash"],
            position=info["position"],
            value=info["value"],
        ),
    )

    while True:
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, truncted, info = environment.step(action)

        record.add(
            dict(
                action=info["action"],
                action_label=info["action_label"],
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

    record.add(
        dict(
            action=info["action"],
            action_label=info["action_label"],
            ret=info["ret"],
            total_profit=info["total_profit"],
        )
    )

    print(record.to_dataframe())

    data = environment.get_data()
    for key, value in data.items():
        print(f"{key}: {value.shape}")