import os
import warnings

from pandas import DataFrame

warnings.filterwarnings("ignore")

import numpy as np
from typing import Any, Dict
import random
import gym
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

from finworld.registry import ENVIRONMENT
from finworld.registry import DATASET
from finworld.utils import TradingRecords
from finworld.trajectory.operation import buy, sell, hold
from finworld.trajectory import max_profit_with_actions_threshold, max_profit_with_actions
from finworld.environment.wrapper import make_env
from finworld.environment.categorical import Categorical
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
                 **kwargs
                 ):
        super(TrajectoryConverter, self).__init__()

        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.level = level
        self.level_format = level_format

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

            feature_ = features.iloc[index - (self.history_timestamps - 1): index + 1].values
            time_ = times.iloc[index - (self.history_timestamps - 1): index + 1].values
            cash_ = cashes.iloc[index - (self.history_timestamps - 1): index + 1].values
            position_ = positions.iloc[index - (self.history_timestamps - 1): index + 1].values

            action_ = actions.iloc[index - (self.history_timestamps - 1): index + 1].values
            ret_ = rets.iloc[index - (self.history_timestamps - 1): index + 1].values
            done_ = dones.iloc[index - (self.history_timestamps - 1): index + 1].values

            item = {
                "features": feature_.astype(np.float32),
                "times": time_.astype(np.int32),
                "cashes": cash_.reshape(-1).astype(np.float32),
                "positions": position_.reshape(-1).astype(np.float32),
                "actions": action_.reshape(-1).astype(np.int32),
                "rets": ret_.reshape(-1).astype(np.float32),
                "dones": done_.reshape(-1).astype(np.float32),
            }
            res_data[timestamp_string] = item

            index += 1

        return res_data


@ENVIRONMENT.register_module(force=True)
class EnvironmentPatchTrading(gym.Env):
    def __init__(self,
                 *args,
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
                 gamma: float = 0.99,
                 use_norm_rewards: bool = True,
                 **kwargs
                 ):
        super(EnvironmentPatchTrading, self).__init__()

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
        self.patch_timestamps = patch_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.max_count_sell = max_count_sell
        self.gamma = gamma
        self.use_norm_rewards = use_norm_rewards

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_df = self.res_info['features_df']
        self.times_df = self.res_info['times_df']
        self.original_prices_df = self.res_info['original_prices_df']

        self.action_labels = ["SELL", "HOLD", "BUY"]  # 0, 1, 2
        self.action_dim = len(self.action_labels)

        self.trajectory_converter = TrajectoryConverter(
            history_timestamps=self.history_timestamps,
            step_timestamps=self.step_timestamps,
            future_timestamps=self.future_timestamps,
            level=self.level,
            level_format=self.level_format,
        )

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
                  position: int,
                  price: float):
        value = cash + position * price
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

    def calculate_expert_values(self, rewards, alpha=0.7, gamma=0.99, use_norm_rewards: bool = True):
        """
        Calculate values for each state with smoothing and normalized rewards.

        :param rewards: List of optimal rewards [r_1^*, r_2^*, ..., r_T^*]
        :param gamma: Discount factor (default=0.9)
        :param alpha: Weight for current reward (default=0.7)
        :return: List of values [V(s_1), V(s_2), ..., V(s_T)]
        """

        rewards = deepcopy(rewards)

        n = len(rewards)
        values = np.zeros(n)

        if use_norm_rewards:
            # Normalize rewards to avoid large variations
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            rewards = (rewards - mean_reward) / (std_reward + 1e-8)

        # Backward recursion to compute values
        for t in range(n - 1, -1, -1):
            if t == n - 1:  # Terminal state
                values[t] = rewards[t]
            else:
                values[t] = alpha * rewards[t] + (1 - alpha) * gamma * values[t + 1]

        return values

    def _init_expert_trajectory(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = self.timestamp_max

        # Load the features and prices
        features_df = self._get_dataitem(self.features_df, start_timestamp, end_timestamp)
        features_timestamps = features_df.index
        times_df = self._get_dataitem(self.times_df, start_timestamp, end_timestamp)

        padding_length = self.history_timestamps - 1

        original_prices_df = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)
        original_prices_df = original_prices_df.iloc[padding_length:]
        price_timestamps = original_prices_df.index

        prices = original_prices_df.values
        close, high, low, open, volume = (prices[:, 0],
                                          prices[:, 1],
                                          prices[:, 2],
                                          prices[:, 3],
                                          prices[:, 4])
        next_close = close[1:]
        next_close = np.append(next_close, close[-1])

        if self.max_count_sell > 0:
            max_profit, actions = max_profit_with_actions_threshold(close,
                                                                    self.initial_amount,
                                                                    self.transaction_cost_pct,
                                                                    self.max_count_sell)
        else:
            max_profit, actions = max_profit_with_actions(close,
                                                          self.initial_amount,
                                                          self.transaction_cost_pct)

        cash = self.initial_amount
        position = 0
        fee_ratio = self.transaction_cost_pct

        records = {
            "cashes": [np.log1p(self.initial_amount)] * padding_length,
            "positions": [.0] * padding_length,
            "actions": [1] * padding_length,
            "rets": [.0] * padding_length,
            "dones": [.0] * padding_length,
        }

        self.expert_evaluate_rewards = {}
        self.expert_actions = {}
        self.expert_logprobs = {}

        value = pre_value = self.initial_amount
        for index, (action_label, price, next_price, timestamp) in enumerate(
                zip(actions, close, next_close, price_timestamps)):

            timestamp_string = timestamp.strftime(self.level_format.value)

            evaluate_reward = self.evaluate_action(cash=cash,
                                                   position=position,
                                                   price=price,
                                                   next_price=next_price,
                                                   value=value)

            self.expert_evaluate_rewards[timestamp_string] = evaluate_reward

            if action_label == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action_label == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = hold(cash, position, price, fee_ratio)

            action = self.action_labels.index(action_label)
            self.expert_actions[timestamp_string] = action

            logits = evaluate_reward.reshape(1, -1)
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            logits = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

            dis = Categorical(logits=logits)
            expert_logprob = dis.log_prob(np.array([action]))[0]
            self.expert_logprobs[timestamp_string] = expert_logprob

            pre_value = cash + position * price
            next_value = cash + position * next_price

            ret = (next_value - pre_value) / pre_value

            if index == len(actions) - 1:
                done = 1
            else:
                done = 0

            records["cashes"].append(np.log1p(float(cash))) # Scale cash with log1p
            records["positions"].append(np.log1p(float(position))) # Scale position with log1p
            records["actions"].append(action)
            records["rets"].append(ret)
            records["dones"].append(done)

        expert_values = self.calculate_expert_values(records["rets"], gamma=self.gamma,
                                                     use_norm_rewards=self.use_norm_rewards)
        self.expert_values = {timestamp.strftime(self.level_format.value): value for timestamp, value in zip(price_timestamps, expert_values)}

        cashes_df = pd.DataFrame(records["cashes"], index=features_timestamps, columns=["cashes"])
        positions_df = pd.DataFrame(records["positions"], index=features_timestamps, columns=["positions"])
        actions_df = pd.DataFrame(records["actions"], index=features_timestamps, columns=["actions"])
        rets_df = pd.DataFrame(records["rets"], index=features_timestamps, columns=["rets"])
        dones_df = pd.DataFrame(records["dones"], index=features_timestamps, columns=["dones"])

        trajectory = {
            "features": features_df,
            "times": times_df,
            "cashes": cashes_df,
            "positions": positions_df,
            "actions": actions_df,
            "rets": rets_df,
            "dones": dones_df,
        }

        return trajectory

    def _init_trading_trajectory(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]

        features_df = self._get_dataitem(self.features_df, start_timestamp, end_timestamp)
        timestamps = features_df.index
        times_df = self._get_dataitem(self.times_df, start_timestamp, end_timestamp)

        nums_timestamps = len(timestamps)

        cashes_df = pd.DataFrame([np.log1p(self.initial_amount)] * nums_timestamps, index=timestamps, columns=["cashes"])
        positions_df = pd.DataFrame([0] * nums_timestamps, index=timestamps, columns=["positions"])

        actions = [1] * nums_timestamps
        rets = [0] * nums_timestamps
        dones = [0] * nums_timestamps
        actions_df = pd.DataFrame(actions, index=timestamps, columns=["actions"])  # set hold as the default action
        rets_df = pd.DataFrame(rets, index=timestamps, columns=["rets"])
        dones_df = pd.DataFrame(dones, index=timestamps, columns=["dones"])  # set 0 as the default done, 1 as the end of the trading

        trajectory = {
            "features": features_df,
            "times": times_df,
            "cashes": cashes_df,
            "positions": positions_df,
            "actions": actions_df,
            "rets": rets_df,
            "dones": dones_df
        }

        return trajectory

    def get_current_trading_data(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]

        features_df = self._get_dataitem(self.features_df, start_timestamp, end_timestamp)
        features_df = features_df.iloc[-1:]
        features = features_df.values
        times_df = self._get_dataitem(self.times_df, start_timestamp, end_timestamp)
        times_df = times_df.iloc[-1:]
        times = times_df.values

        cashes = np.log1p(np.array([self.cash],dtype=np.float32))  # Scale cash with log1p
        positions = np.log1p(np.array([self.position], dtype=np.float32))  # Scale position with log1p

        # add new token
        features = np.concatenate([self.trading_data["features"], features], axis=0)
        times = np.concatenate([self.trading_data["times"], times], axis=0)
        cashes = np.concatenate([self.trading_data["cashes"], cashes], axis=0)
        positions = np.concatenate([self.trading_data["positions"], positions], axis=0)

        # Add the new action and ret
        actions = np.concatenate([self.trading_data["actions"], np.array([self.action])], axis=0)
        rets = np.concatenate([self.trading_data["rets"], np.array([self.ret])], axis=0)
        dones = np.concatenate([self.trading_data["dones"], np.array([1.0 if self.done else 0.0])], axis=0)

        # remove the first token
        features = features[1:]
        times = times[1:]
        cashes = cashes[1:]
        positions = positions[1:]
        actions = actions[1:]
        rets = rets[1:]
        dones = dones[1:]

        trading_data = {
            "features": features.astype(np.float32),
            "times": times.astype(np.int32),
            "cashes": cashes.astype(np.float32),
            "positions": positions.astype(np.float32),
            "actions": actions.astype(np.int32),
            "rets": rets.astype(np.float32),
            "dones": dones.astype(np.float32)
        }

        return trading_data

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
        value = self.get_value(cash=cash, position=position, price=price)

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
        value = self.get_value(cash=cash, position=position, price=price)

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

        value = self.get_value(cash=cash, position=position, price=price)

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

    def evaluate_action(self,
                        cash: float,
                        position: int,
                        price: float,
                        next_price: float,
                        value: float, ):

        rewards = np.zeros(self.action_dim)

        for action in range(self.action_dim):
            amount = action - 1
            if amount > 0:
                res_info = self.buy(cash=cash,
                                    position=position,
                                    price=price,
                                    amount=amount)
            elif amount < 0:
                res_info = self.sell(cash=cash,
                                     position=position,
                                     price=price,
                                     amount=amount)
            else:
                res_info = self.hold(cash=cash,
                                     position=position,
                                     price=price,
                                     amount=amount)

            pre_value = res_info["value"]
            next_value = next_price * res_info["position"] + res_info["cash"]

            reward = (next_value - pre_value) / pre_value

            rewards[action] = reward

        return rewards

    def reset(self, **kwargs):

        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)

        self.state = {}
        self.expert_trajectory = self._init_expert_trajectory()
        self.expert_datas = self.trajectory_converter(self.expert_trajectory)
        self.expert_data = self.expert_datas[self.timestamp_string]

        self.expert_evaluate_reward = self.expert_evaluate_rewards[self.timestamp_string]
        self.expert_value = self.expert_values[self.timestamp_string]
        self.expert_logprob = self.expert_logprobs[self.timestamp_string]

        self.state["expert_trading_cashes"] = self.expert_data["cashes"]
        self.state["expert_trading_positions"] = self.expert_data["positions"]
        self.state["expert_trading_actions"] = self.expert_data["actions"]
        self.state["expert_trading_rets"] = self.expert_data["rets"]

        self.trading_tradjectory = self._init_trading_trajectory()
        self.trading_datas = self.trajectory_converter(self.trading_tradjectory)
        self.trading_data = self.trading_datas[self.timestamp_string]

        self.state["policy_trading_cashes"] = self.trading_data["cashes"]
        self.state["policy_trading_positions"] = self.trading_data["positions"]
        self.state["policy_trading_actions"] = self.trading_data["actions"]
        self.state["policy_trading_rets"] = self.trading_data["rets"]

        self.state["features"] = self.trading_data["features"]
        self.state["times"] = self.trading_data["times"]

        self.price = self.get_price(timestamp_index=self.timestamp_index)

        self.ret = 0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.total_return = 0
        self.total_profit = 0
        self.action = 1
        self.action_label = "HOLD"
        self.expert_action = self.expert_actions[self.timestamp_string]
        self.expert_action_label = self.action_labels[self.expert_action]
        self.done = False

        info = {
            "timestamp": self.timestamp_string,
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "pre_value": self.pre_value,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "action_label": self.action_label,
            "expert_actions": self.expert_actions,
            "expert_action": self.expert_action,
            "expert_action_label": self.expert_action_label,
            "expert_datas": self.expert_datas,
            "expert_data": self.expert_data,
            "expert_evaluate_rewards": self.expert_evaluate_rewards,
            "expert_evaluate_reward": self.expert_evaluate_reward,
            "expert_values": self.expert_values,
            "expert_value": self.expert_value,
            "expert_logprobs": self.expert_logprobs,
            "expert_logprob": self.expert_logprob,
            "done": self.done,
        }

        return self.state, info

    def step(self, action: int = 1):

        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
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

        self.cash = res_info["cash"]
        self.position = res_info["position"]
        self.value = res_info["value"]
        self.action = res_info["action"]
        self.action_label = res_info["action_label"]

        ret = (self.value - self.pre_value) / self.pre_value if self.pre_value != 0 else 0

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
        self.state = {}  # next state

        # Get remaining prices for replanning
        remaining_original_prices_df = self._get_dataitem(self.original_prices_df,
                                                 start_timestamp=self.timestamp_info[self.timestamp_index]["end_timestamp"],
                                                 end_timestamp=self.timestamp_max)
        remaining_prices = remaining_original_prices_df["close"].values if "close" in remaining_original_prices_df.columns else remaining_original_prices_df.values[:, 0]

        self.expert_action = self.expert_actions[self.timestamp_string]
        self.expert_action_label = self.action_labels[self.expert_action]

        self.price = self.get_price(timestamp_index=self.timestamp_index)  # Next price
        self.trading_data = self.get_current_trading_data()
        self.state["policy_trading_cashes"] = self.trading_data["cashes"]
        self.state["policy_trading_positions"] = self.trading_data["positions"]
        self.state["policy_trading_actions"] = self.trading_data["actions"]
        self.state["policy_trading_rets"] = self.trading_data["rets"]
        self.state["features"] = self.trading_data["features"]
        self.state["times"] = self.trading_data["times"]

        # Update expert data dynamically (approximation since full recalculation is costly)
        self.expert_data = self.expert_datas[self.timestamp_string]
        self.state["expert_trading_cashes"] = self.expert_data["cashes"]
        self.state["expert_trading_positions"] = self.expert_data["positions"]
        self.state["expert_trading_actions"] = self.expert_data["actions"]
        self.state["expert_trading_rets"] = self.expert_data["rets"]

        self.expert_evaluate_reward = self.evaluate_action(cash=self.cash,
                                                           position=self.position,
                                                           price=self.price,
                                                           next_price=remaining_prices[1] if len(
                                                               remaining_prices) > 1 else self.price,
                                                           value=self.value)
        logits = self.expert_evaluate_reward.reshape(1, -1)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        logits = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        dis = Categorical(logits=logits)
        self.expert_logprob = dis.log_prob(np.array([self.expert_action]))[0]

        # Approximate expert value (could recompute fully, but here we use precomputed for simplicity)
        self.expert_value = self.expert_values[self.timestamp_string]

        reward = ret

        info = {
            "timestamp": self.timestamp_string,
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "pre_value": self.pre_value,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action,
            "action_label": self.action_label,
            "expert_action": self.expert_action,
            "expert_action_label": self.expert_action_label,
            "expert_data": self.expert_data,
            "expert_evaluate_reward": self.expert_evaluate_reward,
            "expert_value": self.expert_value,
            "expert_logprob": self.expert_logprob,
            "done": self.done,
        }

        # update the pre_value
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncted, info


if __name__ == '__main__':
    symbol = "AAPL"
    num_envs = 4
    history_timestamps = 64
    future_timestamps = 32
    start_timestamp = "2015-05-01"
    end_timestamp = "2025-05-01"
    action_dim = 3
    num_features = 150
    num_times = 4
    num_block_tokens = history_timestamps

    # action-free data
    indicator_transition = ["features"]  # indicator states
    times_transition = ["times"]

    # action-dependent data
    policy_trading_transition = [
        "policy_trading_cashes",
        "policy_trading_positions",
        "policy_trading_actions",
        "policy_trading_rets"
    ]  # history policy states

    expert_trading_transition = [
        "expert_trading_cashes",
        "expert_trading_positions",
        "expert_trading_actions",
        "expert_trading_rets",
    ]

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

    transition = indicator_transition + times_transition + policy_trading_transition + expert_trading_transition + training_transition

    transition_shape = dict(
        features=dict(shape=(num_envs, num_block_tokens, num_features), type="float32", low=-float("inf"), high=float("inf"), obs=True),
        times=dict(shape=(num_envs, num_block_tokens, num_times), type="int32", low=0, high=float("inf"), obs=True),

        policy_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32",  low=-float("inf"), high=float("inf"), obs=True),
        policy_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="float32", low=0, high=float("inf"), obs=True),
        policy_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32", low=0, high=action_dim - 1, obs=True),
        policy_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32", low=-10.0, high=10.0, obs=True),

        expert_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32", low=-float("inf"), high=float("inf"), obs=True),
        expert_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="float32", low=0, high=float("inf"), obs=True),
        expert_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32", low=0, high=action_dim - 1, obs=True),
        expert_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32", low=-10.0, high=10.0, obs=True),

        training_actions=dict(shape=(num_envs,), type="int32", obs=False),
        training_dones=dict(shape=(num_envs,), type="float32", obs=False),
        training_logprobs=dict(shape=(num_envs,), type="float32", obs=False),
        training_rewards=dict(shape=(num_envs,), type="float32", obs=False),
        training_values=dict(shape=(num_envs,), type="float32", obs=False),
        training_advantages=dict(shape=(num_envs,), type="float32", obs=False),
        training_returns=dict(shape=(num_envs,), type="float32", obs=False),
    )

    dataset = dict(
        type="SingleAssetDataset",
        symbol="AAPL",
        data_path="datasets/exp",
        enabled_data_configs = [
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
        if_use_future=False,
        if_norm_temporal=False,
        scaler_cfg = dict(
            type="WindowedScaler"
        ),
        history_timestamps = 64,
        future_timestamps = 32,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level="1day",
    )

    environment = dict(
        type="EnvironmentPatchTrading",
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
        make_env("EnvironmentPatchTrading", env_params=dict(env=deepcopy(environment),
                                               transition_shape=transition_shape, seed=2024 + i)) for
        i in range(num_envs)
    ])

    records = [TradingRecords() for _ in range(num_envs)]

    state, info = environments.reset()
    expert_actions = info["expert_actions"][0]

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
        "logprob:", np.mean([item for item in info["expert_logprob"]]),
        "value:", np.mean([item for item in info["expert_value"]]),
        "expert_action:", info["expert_action"]
    )

    for i in range(500):
        action = info["expert_action"]
        next_state, reward, done, truncted, info = environments.step(action)

        for i in range(num_envs):
            record = records[i]
            record.add(
                dict(
                    action=info["action"][i],
                    action_label=info["action_label"][i],
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
            "logprob:", np.mean([item for item in info["expert_logprob"]]),
            "value:", np.mean([item for item in info["expert_value"]]),
            "expert_action:", info["expert_action"]
        )
        if "final_info" in info:
            break

    for i in range(num_envs):
        record = records[i]
        record.add(
            dict(
                action=info["action"][i],
                action_label=info["action_label"][i],
                ret=info["ret"][i],
                total_profit=info["total_profit"][i],
            )
        )
    environments.close()

    print([
       records[i].to_dataframe() for i in range(num_envs)
    ])