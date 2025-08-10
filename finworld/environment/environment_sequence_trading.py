import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from typing import Any, Dict, Union, List
import random
import gym
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

from finworld.registry import ENVIRONMENT
from finworld.registry import DATASET
from finworld.utils import assemble_project_path
from finworld.trajectory.operation import buy, sell, hold
from finworld.trajectory import max_profit_with_actions_threshold, max_profit_with_actions
from finworld.environment.wrapper import make_env

class TrajectoryConverter():
    def __init__(self,
                 *args,
                 history_timestamps: int = 35,  # history_timestamps = 32 + (4 - 1) = 35
                 patch_timestamps: int = 4,
                 step_timestamps: int = 1,
                 future_timestamps: int = 32,
                 **kwargs
                 ):
        super(TrajectoryConverter, self).__init__()

        self.history_timestamps = history_timestamps
        self.patch_timestamps = patch_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps

        self.num_patches = (self.history_timestamps - self.patch_timestamps) // self.step_timestamps + 1

    def __call__(self, trajectory: Dict[str, Any]):
        features = trajectory["features"]
        cashes = trajectory["cashes"]
        positions = trajectory["positions"]
        actions = trajectory["actions"]
        rets = trajectory["rets"]
        dones = trajectory["dones"]

        timestamps = features.index
        index = self.history_timestamps - 1
        timestamps = timestamps[index:]

        res_data = OrderedDict()

        for timestamp in timestamps:

            feature = features.iloc[index - (self.history_timestamps - 1): index + 1].values
            cash = cashes.iloc[index - (self.history_timestamps - 1): index + 1].values
            position = positions.iloc[index - (self.history_timestamps - 1): index + 1].values

            action = actions.iloc[index - (self.history_timestamps - 1): index + 1].values
            ret = rets.iloc[index - (self.history_timestamps - 1): index + 1].values
            done = dones.iloc[index - (self.history_timestamps - 1): index + 1].values

            patch_feature = np.array([feature[i: i + self.patch_timestamps] for i in range(0, self.num_patches, self.step_timestamps)])
            patch_cash = cash[self.patch_timestamps - 1:, :]
            patch_position = position[self.patch_timestamps - 1:, :]
            patch_action = action[self.patch_timestamps - 1:, :]
            patch_ret = ret[self.patch_timestamps - 1:, :]
            patch_done = done[self.patch_timestamps - 1:, :]

            item = {
                "features": patch_feature,
                "cashes": patch_cash.reshape(-1),
                "positions": patch_position.reshape(-1),
                "actions": patch_action.reshape(-1),
                "rets": patch_ret.reshape(-1),
                "dones": patch_done.reshape(-1)
            }

            res_data[timestamp] = item

            index += 1

        return res_data


@ENVIRONMENT.register_module(force=True)
class EnvironmentSequenceTrading(gym.Env):
    def __init__(self,
                 *args,
                 mode: str = "train",
                 dataset: Any = None,
                 select_asset: str = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 timestamp_format: str = "%Y-%m-%d",
                 history_timestamps: int = 35, # history_timestamps = 32 + (4 - 1) = 35
                 patch_timestamps: int = 4,
                 step_timestamps: int = 1,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 max_count_sell: int = 1,
                 position_num_bins: int = 1000,
                 position_max_value: int = int(1e6),
                 **kwargs
                 ):
        super(EnvironmentSequenceTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.select_asset = select_asset
        assert self.select_asset in self.dataset.assets and self.select_asset is not None, \
            f"select_asset {self.select_asset} not in assets {self.dataset.assets}"

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.assets = self.dataset.assets
        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.patch_timestamps = patch_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format
        self.max_count_sell = max_count_sell
        self.position_num_bins = position_num_bins
        self.position_max_value = position_max_value

        self.num_patches = (self.history_timestamps - self.patch_timestamps) // self.step_timestamps + 1

        self.timestamp_info, self.features_df, self.prices_df = self._init_features()

        self.action_labels = ["SELL", "HOLD", "BUY"]  # 0, 1, 2
        self.action_dim = len(self.action_labels)

        self.trajectory_converter = TrajectoryConverter(
            history_timestamps=self.history_timestamps,
            patch_timestamps=self.patch_timestamps,
            step_timestamps=self.step_timestamps,
            future_timestamps=self.future_timestamps
        )

        self.PLACEHOLDER = 0

    def _init_features(self):

        timestamp_info = {}
        for key, value in self.dataset.data_info.items():
            start_timestamp = value["history_info"]["start_timestamp"]
            end_timestamp = value["history_info"]["end_timestamp"]

            if end_timestamp >= pd.to_datetime(self.start_timestamp) and end_timestamp < pd.to_datetime(self.end_timestamp):
                timestamp_info[key] = {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }

        self.timestamp_min_index = min(timestamp_info.keys())
        self.timestamp_max_index = max(timestamp_info.keys())
        self.timestamp_min = timestamp_info[self.timestamp_min_index]["end_timestamp"]
        self.timestamp_max = timestamp_info[self.timestamp_max_index]["end_timestamp"]

        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1
        assert self.num_timestamps == len(timestamp_info), f"num_timestamps {self.num_timestamps} != len(data_info) {len(timestamp_info)}"

        features = self.dataset.features
        prices = self.dataset.original_prices

        features = features[self.select_asset]
        prices = prices[self.select_asset]

        features_df = features
        prices_df = prices

        return timestamp_info, features_df, prices_df


    def _init_timestamp_index(self):
        if self.mode == "train":
            timestamp_index = random.randint(self.timestamp_min_index, self.timestamp_min_index + 3 * (self.num_timestamps // 4))
        else:
            timestamp_index = self.timestamp_min_index
        return timestamp_index

    def get_timestamp(self, timestamp_index: int):
        return self.timestamp_info[timestamp_index]["end_timestamp"]

    def get_value(self,
                  cash: float,
                  postition: int,
                  price: float):
        value = cash + postition * price
        return value

    def get_price(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        end_timestamp = timestamp_info["end_timestamp"]

        prices = self.prices_df.loc[end_timestamp].values

        o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
        price = adj

        return price

    def _init_expert_trajectory(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = self.timestamp_max

        # Load the features and prices
        features_df = self.features_df.loc[start_timestamp:end_timestamp]
        features_timestamps = features_df.index

        padding_length = self.patch_timestamps - 1
        prices_df = self.prices_df.loc[start_timestamp:end_timestamp]
        prices_df = prices_df.iloc[padding_length:]
        price_timestamps = prices_df.index

        prices = prices_df.values
        o, h, l, c, adj = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3], prices[:, 4]

        if self.max_count_sell > 0:
            max_profit, actions = max_profit_with_actions_threshold(adj,
                                                                    self.initial_amount,
                                                                    self.transaction_cost_pct,
                                                                    self.max_count_sell)
        else:
            max_profit, actions = max_profit_with_actions(adj,
                                                          self.initial_amount,
                                                          self.transaction_cost_pct)
        cash = self.initial_amount
        position = 0
        fee_ratio = self.transaction_cost_pct
        value = cash

        records = {
            "cashes": [np.log1p(self.initial_amount)] * padding_length,
            "positions": [0] * padding_length,
            "actions": [1] * padding_length,
            "rets": [0] * padding_length,
            "dones": [0] * padding_length,
        }

        self.expert_actions = {}
        self.expert_evaluate_rewards = {}
        for index, (action_label, price, timestamp) in enumerate(zip(actions, adj, price_timestamps)):

            evaluate_reward = self.evaluate_action(cash=cash,
                                                   position=position,
                                                   price=price,
                                                   value=value)
            self.expert_evaluate_rewards[timestamp] = evaluate_reward

            if action_label == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action_label == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = hold(cash, position, price, fee_ratio)

            action = self.action_labels.index(action_label)
            self.expert_actions[timestamp] = action

            value_ = cash + position * price
            ret = (value_ - value) / value
            value = value_

            if index == len(actions) - 1:
                done = 1
            else:
                done = 0

            records["cashes"].append(cash)
            records["positions"].append(position)
            records["actions"].append(action)
            records["rets"].append(ret)
            records["dones"].append(done)

        cashes_df = pd.DataFrame(records["cashes"], index=features_timestamps, columns=["cashes"])
        binned_positions = np.clip(np.floor(np.array(records["positions"]) / self.position_max_value * self.position_num_bins),
                                   0, self.position_num_bins - 1).astype(int)
        positions_df = pd.DataFrame(binned_positions, index=features_timestamps, columns=["positions"])
        actions_df = pd.DataFrame(records["actions"], index=features_timestamps, columns=["actions"])
        rets_df = pd.DataFrame(records["rets"], index=features_timestamps, columns=["rets"])
        dones_df = pd.DataFrame(records["dones"], index=features_timestamps, columns=["dones"])

        trajectory = {
            "features": features_df,
            "cashes": cashes_df,
            "positions": positions_df,
            "actions": actions_df,
            "rets": rets_df,
            "dones": dones_df
        }

        return trajectory

    def _init_trading_trajectory(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]

        features_df = self.features_df.loc[start_timestamp:end_timestamp]
        timestamps = features_df.index

        nums_timestamps = len(timestamps)

        cashes_df = pd.DataFrame([np.log1p(self.initial_amount)] * nums_timestamps, index=timestamps, columns=["cashes"])
        positions_df = pd.DataFrame([0] * nums_timestamps, index=timestamps, columns=["positions"])

        actions = [1] * nums_timestamps
        actions[-1] = self.PLACEHOLDER # Set the last action as the placeholder, which will be replaced by the true action
        rets = [0] * nums_timestamps
        rets[-1] = self.PLACEHOLDER # Set the last ret as the placeholder, which will be replaced by the true ret
        dones = [0] * nums_timestamps
        dones[-1] = self.PLACEHOLDER # Set the last done as 1, which will be replaced by the true done
        actions_df = pd.DataFrame(actions, index=timestamps, columns=["actions"]) # set hold as the default action
        rets_df = pd.DataFrame(rets, index=timestamps, columns=["rets"])
        dones_df = pd.DataFrame(dones, index=timestamps, columns=["dones"]) # set 0 as the default done, 1 as the end of the trading


        trajectory = {
            "features": features_df,
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

        features_df = self.features_df.loc[start_timestamp:end_timestamp]
        features_df = features_df.iloc[-self.patch_timestamps:]
        features = features_df.values
        features = features[np.newaxis, ...]

        cashes = np.log1p(np.array([self.cash]))
        positions = np.array([self.position])
        binned_positions = np.clip(np.floor(positions / self.position_max_value * self.position_num_bins), 0, self.position_num_bins - 1).astype(int)
        positions = binned_positions

        # add new token
        features = np.concatenate([self.trading_data["features"], features], axis=0)
        cashes = np.concatenate([self.trading_data["cashes"], cashes], axis=0)
        positions = np.concatenate([self.trading_data["positions"], positions], axis=0)
        actions = np.insert(self.trading_data["actions"], -1, self.action) # Replace the last action with the true action
        rets = np.insert(self.trading_data["rets"], -1, self.ret) # Replace the last ret with the true ret
        dones = np.insert(self.trading_data["dones"], -1, 0 if self.done is False else 1) # Replace the last done with the true done

        # remove the first token
        features = features[1:]
        cashes = cashes[1:]
        positions = positions[1:]
        actions = actions[1:]
        rets = rets[1:]
        dones = dones[1:]

        trading_data = {
            "features": features,
            "cashes": cashes,
            "positions": positions,
            "actions": actions,
            "rets": rets,
            "dones": dones
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

    def evaluate_action(self,
                        cash: float,
                        position: int,
                        price: float,
                        value: float,):
        pre_value = value

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

            post_value = res_info["value"]
            reward = (post_value - pre_value) / pre_value

            rewards[action] = reward

        return rewards

    def reset(self, **kwargs):

        self.timestamp_index = self._init_timestamp_index()
        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)

        self.state = {}
        self.expert_trajectory = self._init_expert_trajectory()
        self.expert_datas = self.trajectory_converter(self.expert_trajectory)
        self.expert_data = self.expert_datas[self.timestamp]
        self.expert_evaluate_reward = self.expert_evaluate_rewards[self.timestamp]

        self.state["expert_trading_cashes"] = self.expert_data["cashes"]
        self.state["expert_trading_positions"] = self.expert_data["positions"]
        self.state["expert_trading_actions"] = self.expert_data["actions"]
        self.state["expert_trading_rets"] = self.expert_data["rets"]

        self.trading_tradjectory = self._init_trading_trajectory()
        self.trading_datas = self.trajectory_converter(self.trading_tradjectory)
        self.trading_data = self.trading_datas[self.timestamp]

        self.state["policy_trading_cashes"] = self.trading_data["cashes"]
        self.state["policy_trading_positions"] = self.trading_data["positions"]
        self.state["policy_trading_actions"] = self.trading_data["actions"]
        self.state["policy_trading_rets"] = self.trading_data["rets"]

        self.state["features"] = self.trading_data["features"]

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
        self.expert_action = self.expert_actions[self.timestamp]
        self.expert_action_label = self.action_labels[self.expert_action]
        self.done = False
        self.evaluate_rewards = self.evaluate_action(cash=self.cash,
                                                     position=self.position,
                                                     price=self.price,
                                                     value=self.value)

        info= {
            "timestamp": self.timestamp.strftime(self.timestamp_format),
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
            "expert_datas": self.expert_datas,
            "expert_data": self.expert_data,
            "expert_evaluate_rewards": self.expert_evaluate_rewards,
            "expert_evaluate_reward": self.expert_evaluate_reward,
            "done": self.done,
        }

        return self.state, info

    def step(self, action: int = 1):

        action = action - 1

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

        ret = (self.value - self.pre_value) / self.pre_value
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
        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)
        self.state = {}
        self.expert_action = self.expert_actions[self.timestamp]
        self.expert_action_label = self.action_labels[self.expert_action]
        self.expert_data = self.expert_datas[self.timestamp]
        self.state["expert_trading_cashes"] = self.expert_data["cashes"]
        self.state["expert_trading_positions"] = self.expert_data["positions"]
        self.state["expert_trading_actions"] = self.expert_data["actions"]
        self.state["expert_trading_rets"] = self.expert_data["rets"]
        self.price = self.get_price(timestamp_index=self.timestamp_index) # next price
        self.trading_data = self.get_current_trading_data()
        self.state["policy_trading_cashes"] = self.trading_data["cashes"]
        self.state["policy_trading_positions"] = self.trading_data["positions"]
        self.state["policy_trading_actions"] = self.trading_data["actions"]
        self.state["policy_trading_rets"] = self.trading_data["rets"]
        self.state["features"] = self.trading_data["features"]
        self.expert_evaluate_reward = self.expert_evaluate_rewards[self.timestamp]

        reward = ret

        info = {
            "timestamp": self.timestamp.strftime(self.timestamp_format),
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
            "done": self.done,
        }

        # update the pre_value
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncted, info

if __name__ == '__main__':
    select_asset = "AAPL"
    num_envs = 2
    num_block_tokens = 32
    action_dim = 3
    patch_timestamps = 4
    num_features = 152

    # action-free data
    indicator_transition = ["features"]  # indicator states

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

    transition = indicator_transition + policy_trading_transition + expert_trading_transition + training_transition

    transition_shape = dict(
        features=dict(shape=(num_envs, num_block_tokens, patch_timestamps, num_features), type="float32",
                      low=-float("inf"), high=float("inf"), obs=True),

        policy_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32",
                                   low=-float("inf"), high=float("inf"), obs=True),
        policy_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="int32",
                                      low=0, high=float("inf"), obs=True),
        policy_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32",
                                    low=0, high=action_dim - 1, obs=True),
        policy_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32",
                                 low=-10.0, high=10.0, obs=True),

        expert_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32",
                                   low=-float("inf"), high=float("inf"), obs=True),
        expert_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="int32",
                                      low=0, high=float("inf"), obs=True),
        expert_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32",
                                    low=0, high=action_dim - 1, obs=True),
        expert_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32",
                                 low=-10.0, high=10.0, obs=True),

        training_actions=dict(shape=(num_envs,), type="int32", obs=False),
        training_dones=dict(shape=(num_envs,), type="float32", obs=False),
        training_logprobs=dict(shape=(num_envs,), type="float32", obs=False),
        training_rewards=dict(shape=(num_envs,), type="float32", obs=False),
        training_values=dict(shape=(num_envs,), type="float32", obs=False),
        training_advantages=dict(shape=(num_envs,), type="float32", obs=False),
        training_returns=dict(shape=(num_envs,), type="float32", obs=False),
    )

    dataset = dict(
        type="MultiAssetDataset",
        data_path="datasets/processd_day_dj30/features",
        assets_path="configs/_asset_list_/dj30.json",
        fields_name={
            "features": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "kmid",
                "kmid2",
                "klen",
                "kup",
                "kup2",
                "klow",
                "klow2",
                "ksft",
                "ksft2",
                "roc_5",
                "roc_10",
                "roc_20",
                "roc_30",
                "roc_60",
                "ma_5",
                "ma_10",
                "ma_20",
                "ma_30",
                "ma_60",
                "std_5",
                "std_10",
                "std_20",
                "std_30",
                "std_60",
                "beta_5",
                "beta_10",
                "beta_20",
                "beta_30",
                "beta_60",
                "max_5",
                "max_10",
                "max_20",
                "max_30",
                "max_60",
                "min_5",
                "min_10",
                "min_20",
                "min_30",
                "min_60",
                "qtlu_5",
                "qtlu_10",
                "qtlu_20",
                "qtlu_30",
                "qtlu_60",
                "qtld_5",
                "qtld_10",
                "qtld_20",
                "qtld_30",
                "qtld_60",
                "rank_5",
                "rank_10",
                "rank_20",
                "rank_30",
                "rank_60",
                "imax_5",
                "imax_10",
                "imax_20",
                "imax_30",
                "imax_60",
                "imin_5",
                "imin_10",
                "imin_20",
                "imin_30",
                "imin_60",
                "imxd_5",
                "imxd_10",
                "imxd_20",
                "imxd_30",
                "imxd_60",
                "rsv_5",
                "rsv_10",
                "rsv_20",
                "rsv_30",
                "rsv_60",
                "cntp_5",
                "cntp_10",
                "cntp_20",
                "cntp_30",
                "cntp_60",
                "cntn_5",
                "cntn_10",
                "cntn_20",
                "cntn_30",
                "cntn_60",
                "cntd_5",
                "cntd_10",
                "cntd_20",
                "cntd_30",
                "cntd_60",
                "corr_5",
                "corr_10",
                "corr_20",
                "corr_30",
                "corr_60",
                "cord_5",
                "cord_10",
                "cord_20",
                "cord_30",
                "cord_60",
                "sump_5",
                "sump_10",
                "sump_20",
                "sump_30",
                "sump_60",
                "sumn_5",
                "sumn_10",
                "sumn_20",
                "sumn_30",
                "sumn_60",
                "sumd_5",
                "sumd_10",
                "sumd_20",
                "sumd_30",
                "sumd_60",
                "vma_5",
                "vma_10",
                "vma_20",
                "vma_30",
                "vma_60",
                "vstd_5",
                "vstd_10",
                "vstd_20",
                "vstd_30",
                "vstd_60",
                "wvma_5",
                "wvma_10",
                "wvma_20",
                "wvma_30",
                "wvma_60",
                "vsump_5",
                "vsump_10",
                "vsump_20",
                "vsump_30",
                "vsump_60",
                "vsumn_5",
                "vsumn_10",
                "vsumn_20",
                "vsumn_30",
                "vsumn_60",
                "vsumd_5",
                "vsumd_10",
                "vsumd_20",
                "vsumd_30",
                "vsumd_60",
            ],
            "prices": [
                "open",
                "high",
                "low",
                "close",
                "adj_close",
            ],
            "temporals": [
                "day",
                "weekday",
                "month",
            ],
            "labels": [
                "ret1",
                "mov1"
            ]
        },
        if_norm=True,
        if_norm_temporal=False,
        if_use_future=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        scaler_file="scalers.joblib",
        scaled_data_file="scaled_data.joblib",
        history_timestamps=35,
        future_timestamps=32,
        start_timestamp="2008-04-01",
        end_timestamp="2024-06-01",
        timestamp_format="%Y-%m-%d",
        exp_path=assemble_project_path(os.path.join("workdir", "sequence_trading_testing"))
    )

    environment = dict(
        type="EnvironmentSequenceTrading",
        mode="train",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        timestamp_format="%Y-%m-%d",
        start_timestamp="2008-04-01",
        end_timestamp="2024-06-01",
        history_timestamps=35,
        patch_timestamps=4,
        step_timestamps=1,
        future_timestamps=32,
    )

    dataset = DATASET.build(dataset)

    environment.update({
        "dataset": dataset,
        "select_asset": select_asset
    })

    environment = ENVIRONMENT.build(environment)

    environments = gym.vector.AsyncVectorEnv([
        make_env("Trading-v1", env_params=dict(env=deepcopy(environment),
                                               transition_shape=transition_shape, seed=2024 + i)) for
        i in range(num_envs)
    ])

    state, info = environments.reset()

    for key, value in state.items():
        print(f"{key}: {value.shape}")
    print("timestamp:", info["timestamp"], "total_profit:", info["total_profit"], info["evaluate_rewards"])

    for i in range(100):
        action = [1] * num_envs
        next_state, reward, done, truncted, info = environments.step(action)
        for key, value in next_state.items():
            print(f"{key}: {value.shape}")
        print("timestamp:", info["timestamp"], "total_profit:", info["total_profit"], info["evaluate_rewards"])
        if "final_info" in info:
            break
    environments.close()