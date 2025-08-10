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

class TrajectoryConverter():
    def __init__(self,
                 *args,
                 history_timestamps: int = 32,  # history_timestamps = 32 + (4 - 1) = 35
                 step_timestamps: int = 1,
                 future_timestamps: int = 32,
                 level: TimeLevel = TimeLevel.DAY,
                 level_format: TimeLevelFormat = TimeLevelFormat.DAY,
                 symbols: List[str] = [],
                 **kwargs
                 ):
        super(TrajectoryConverter, self).__init__()

        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.level = level
        self.level_format = level_format
        self.symbols = symbols

    def __call__(self, trajectory: Dict[str, Any]):
        
        features = trajectory["features"]
        times = trajectory["times"]
        cashes = trajectory["cashes"]
        positions = trajectory["positions"]
        actions = trajectory["actions"]
        rets = trajectory["rets"]
        dones = trajectory["dones"]

        timestamps = times.index
        index = self.history_timestamps - 1
        timestamps = timestamps[index:]

        res_data = OrderedDict()

        for timestamp in timestamps:
            timestamp_string = timestamp.strftime(self.level_format.value)
            
            features_ = []
            for symbol in self.symbols:
                feature_ = features[symbol].iloc[index - (self.history_timestamps - 1): index + 1].values
                features_.append(feature_)
            features_ = np.array(features_).transpose(1, 0, 2)
            times_ = times.iloc[index - (self.history_timestamps - 1): index + 1].values
            cashs_ = cashes.iloc[index - (self.history_timestamps - 1): index + 1].values
            positions_ = positions.iloc[index - (self.history_timestamps - 1): index + 1].values
            
            actions_ = actions.iloc[index - (self.history_timestamps - 1): index + 1].values
            rets_ = rets.iloc[index - (self.history_timestamps - 1): index + 1].values
            dones_ = dones.iloc[index - (self.history_timestamps - 1): index + 1].values

            item = {
                "features": features_.astype(np.float32),
                "times": times_.astype(np.int32),
                "cashes": cashs_.reshape(-1).astype(np.float32),
                "positions": positions_.astype(np.float32),
                "actions": actions_.astype(np.float32),
                "rets": rets_.reshape(-1).astype(np.float32),
                "dones": dones_.reshape(-1).astype(np.float32),
            }
            res_data[timestamp_string] = item

            index += 1

        return res_data

@ENVIRONMENT.register_module(force=True)
class EnvironmentPatchPortfolio(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 history_timestamps: int = 32,
                 patch_timestamps: int = 4,
                 step_timestamps: int = 1,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 max_count_sell: int = -1,
                 position_num_bins: int = 1000,
                 position_max_value: int = int(1e6),
                 gamma: float = 0.99,
                 use_norm_rewards: bool = True,
                 **kwargs
                 ):
        super(EnvironmentPatchPortfolio, self).__init__()
        
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
        self.max_count_sell = max_count_sell
        self.position_num_bins = position_num_bins
        self.position_max_value = position_max_value
        self.gamma = gamma
        self.use_norm_rewards = use_norm_rewards
        self.action_dim = 1 + len(self.symbols)  # cash + assets

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_dfs = self.res_info['features_dfs']
        self.times_dfs = self.res_info['times_dfs']
        self.original_prices_dfs = self.res_info['original_prices_dfs']
        
        self.trajectory_converter = TrajectoryConverter(
            history_timestamps=self.history_timestamps,
            step_timestamps=self.step_timestamps,
            future_timestamps=self.future_timestamps,
            level=self.level,
            level_format=self.level_format,
            symbols=self.symbols
        )
        
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
            times_dfs=times_dfs,
            original_prices_dfs=original_prices_dfs,
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
    
    def _init_portfolio_trajectory(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        
        features_dfs = {}
        for symbol in self.symbols:
            features_dfs[symbol] = self._get_dataitem(self.features_dfs[symbol], start_timestamp, end_timestamp)
        times_df = self._get_dataitem(self.times_dfs[self.symbols[0]], start_timestamp, end_timestamp)
        timestamps = times_df.index
        
        cashes_df = pd.DataFrame(columns=["cash"], index=timestamps)
        actions_df = pd.DataFrame(columns=["cash"] + [symbol for symbol in self.symbols], index=timestamps)
        positions_df = pd.DataFrame(columns=[symbol for symbol in self.symbols], index=timestamps)
        rets_df = pd.DataFrame(columns=["ret"], index=timestamps)
        dones_df = pd.DataFrame(columns=["done"], index=timestamps)
        
        for timestamp in timestamps:
            cash = np.log1p(self.initial_amount)
            cashes_df.loc[timestamp] = cash
            action = {
                "cash": 1.0,
                **{symbol: 0.0 for symbol in self.symbols}
            }
            actions_df.loc[timestamp] = action
            position = {
                **{symbol: 0.0 for symbol in self.symbols}
            }
            positions_df.loc[timestamp] = position
            rets_df.loc[timestamp] = 0.0
            dones_df.loc[timestamp] = 0

        trajectory = {
            "features": features_dfs,
            "times": times_df,
            "cashes": cashes_df,
            "actions": actions_df,
            "positions": positions_df,
            "rets": rets_df,
            "dones": dones_df
        }

        return trajectory
    
    def get_current_portfolio_data(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        
        features = []
        for symbol in self.symbols:
            feature_df = self._get_dataitem(self.features_dfs[symbol], start_timestamp, end_timestamp)
            feature = feature_df.iloc[-1:]
            features.append(feature)
        features = np.array(features).transpose(1, 0, 2)
        
        times = self._get_dataitem(self.times_dfs[self.symbols[0]], start_timestamp, end_timestamp)
        times = times.iloc[-1:]
        times = times.values
        
        cashes = np.log1p(np.array([self.cash]))
        positions = np.array([self.position])
        actions = np.array([self.action])
        rets = np.array([self.ret])
        dones = np.array([0 if self.done else 1])
        
        features = np.concatenate([self.portfolio_data["features"], features], axis=0)
        times = np.concatenate([self.portfolio_data["times"], times], axis=0)
        cashes = np.concatenate([self.portfolio_data["cashes"], cashes], axis=0)
        positions = np.concatenate([self.portfolio_data["positions"], positions], axis=0)
        actions = np.concatenate([self.portfolio_data["actions"], actions], axis=0)
        rets = np.concatenate([self.portfolio_data["rets"], rets], axis=0)
        dones = np.concatenate([self.portfolio_data["dones"], dones], axis=0)
        
        # remove the first token
        features = features[1:]
        times = times[1:]
        cashes = cashes[1:]
        positions = positions[1:]
        actions = actions[1:]
        rets = rets[1:]
        dones = dones[1:]
        
        portfolio_data = {
            "features": features.astype(np.float32),
            "times": times.astype(np.int32),
            "cashes": cashes.astype(np.float32),
            "positions": positions.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rets": rets.astype(np.float32),
            "dones": dones.astype(np.float32),
        }
        
        return portfolio_data
    
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

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)

        self.state = {}
        
        self.portfolio_trajectory = self._init_portfolio_trajectory()
        self.portfolio_datas = self.trajectory_converter(self.portfolio_trajectory)
        self.portfolio_data = self.portfolio_datas[self.timestamp_string]
        
        self.state["policy_portfolio_cashes"] = self.portfolio_data["cashes"]
        self.state["policy_portfolio_positions"] = self.portfolio_data["positions"]
        self.state["policy_portfolio_actions"] = self.portfolio_data["actions"]
        self.state["policy_portfolio_rets"] = self.portfolio_data["rets"]
        
        self.state["features"] = self.portfolio_data["features"]
        self.state["times"] = self.portfolio_data["times"]
        
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
        
        self.state = {}
        
        self.price = self.get_price(timestamp_index=self.timestamp_index)
        self.portfolio_data = self.get_current_portfolio_data()
        self.state["policy_portfolio_cashes"] = self.portfolio_data["cashes"]
        self.state["policy_portfolio_positions"] = self.portfolio_data["positions"]
        self.state["policy_portfolio_actions"] = self.portfolio_data["actions"]
        self.state["policy_portfolio_rets"] = self.portfolio_data["rets"]
        self.state["features"] = self.portfolio_data["features"]
        self.state["times"] = self.portfolio_data["times"]
        
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
    start_timestamp = "2015-05-01"
    end_timestamp = "2025-05-01"
    action_dim = 6 + 1
    num_features = 150
    num_times = 4
    num_assets = 6
    num_block_tokens = history_timestamps

    # action-free data
    indicator_transition = ["features"]  # indicator states
    times_transition = ["times"]
    
    # action-dependent data
    portfolio_transition = [
        "policy_portfolio_cashes",
        "policy_portfolio_positions",
        "policy_portfolio_actions",
        "policy_portfolio_rets"
    ]  # history policy states

    # training data
    training_transition = [
        "training_actions",
        "training_dones",
        "training_logprobs",
        "training_rewards",
        "training_values",
        "training_advantages",
        "training_returns"
    ]

    transition = indicator_transition + times_transition + portfolio_transition + training_transition

    transition_shape = dict(
        features=dict(shape=(num_envs, num_block_tokens, num_assets, num_features), type="float32", low=-float("inf"), high=float("inf"), obs=True),
        times=dict(shape=(num_envs, num_block_tokens, num_times), type="int32", low=0, high=float("inf"), obs=True),
        
        policy_portfolio_cashes=dict(shape=(num_envs, num_block_tokens), type="float32", low=0, high=float("inf"), obs=True),
        policy_portfolio_positions=dict(shape=(num_envs, num_block_tokens, num_assets), type="float32", low=0, high=float("inf"), obs=True),
        policy_portfolio_actions=dict(shape=(num_envs, num_block_tokens, 1 + num_assets), type="float32", low=0, high= 1.0, obs=True),
        policy_portfolio_rets=dict(shape=(num_envs, num_block_tokens), type="float32", low=-float("inf"), high=float("inf"), obs=True),
        
        training_actions=dict(shape=(num_envs, 1 + num_assets), type="float32", obs=False),
        training_dones=dict(shape=(num_envs, ), type="float32", obs=False),
        training_logprobs=dict(shape=(num_envs, ), type="float32", obs=False),
        training_rewards=dict(shape=(num_envs, ), type="float32", obs=False),
        training_values=dict(shape=(num_envs, ), type="float32", obs=False),
        training_advantages=dict(shape=(num_envs, ), type="float32", obs=False),
        training_returns=dict(shape=(num_envs, ), type="float32", obs=False),
    )
    
    dataset = dict(
        type="MultiAssetDataset",
        assets_name="exp",
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
        end_timestamp="2025-05-01",
        level="1day",
    )
    
    environment = dict(
        type="EnvironmentPatchPortfolio",
        mode="train",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        patch_timestamps=4,
        step_timestamps=1,
    )

    dataset = DATASET.build(dataset)

    environment.update({
        "dataset": dataset
    })

    environment = ENVIRONMENT.build(environment)

    environments = gym.vector.AsyncVectorEnv([
        make_env("EnvironmentPatchPortfolio", env_params=dict(env=deepcopy(environment), transition_shape=transition_shape, seed=2024 + i)) for
        i in range(num_envs)
    ])

    records = [PortfolioRecords() for _ in range(num_envs)]

    state, info = environments.reset()
    for i in range(num_envs):
        record = records[i]
        record.add(
            dict(
                timestamp=info["timestamp"][i],
                price=info["price"][i],
                cash=info["cash"][i],
                position=info["position"][i],
                value=info["value"][i],
            ),
        )

    for key, value in state.items():
        print(f"{key}: {value.shape}")
    print(
        "timestamp:", info["timestamp"],
        "total_profit:", info["total_profit"],
    )

    for i in range(500):

        action = []
        for i in range(num_envs):
            action_item = np.abs(np.random.randn(action_dim).astype(np.float32))
            action_item = action_item / np.sum(action_item)
            action.append(action_item)

        next_state, reward, done, truncted, info = environments.step(action)

        for i in range(num_envs):
            record = records[i]
            record.add(
                dict(
                    action=info["action"][i],
                    ret=info["ret"][i],
                    total_profit=info["total_profit"][i],
                    timestamp=info["timestamp"][i],  # next timestamp
                    price=info["price"][i],  # next price
                    cash=info["cash"][i],  # next cash
                    position=info["position"][i],  # next position
                    value=info["value"][i],  # next value
                ),
            )

        for key, value in next_state.items():
            print(f"{key}: {value.shape}")
        print(
            "timestamp:", info["timestamp"],
            "total_profit:", info["total_profit"],
        )
        if "final_info" in info:
            break

    for i in range(num_envs):
        record = records[i]
        record.add(
            dict(
                action=info["action"][i],
                ret=info["ret"][i],
                total_profit=info["total_profit"][i],
            )
        )
    environments.close()

    print([
        records[i].to_dataframe() for i in range(num_envs)
    ])