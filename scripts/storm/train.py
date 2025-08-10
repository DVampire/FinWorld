import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from mmengine import DictAction
from copy import deepcopy
import pathlib
import time
from dotenv import load_dotenv
import torch.optim as optim
from einops import rearrange
from typing import Dict, List, Any
import pandas as pd
import gym
from gym import Wrapper, spaces
import random
from gym.envs.registration import register
from typing import List, Any
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import Mlp
from torch.distributions.categorical import Categorical

load_dotenv(verbose=True)

root = str(pathlib.Path(__file__).resolve().parents[1])
current = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(current)
sys.path.append(root)

from finworld.config import build_config
from finworld.log import logger
from finworld.log import tensorboard_logger
from finworld.log import wandb_logger
from finworld.metrics import ARR, SR, MDD, CR, SOR, VOL
from finworld.utils import assemble_project_path
from finworld.utils import to_torch_dtype
from finworld.utils import save_json, load_json
from finworld.utils import load_joblib

def build_storage(shape, type, device):
    if type.startswith("int32"):
        type = torch.int32
    elif type.startswith("float32"):
        type = torch.float32
    elif type.startswith("int64"):
        type = torch.int64
    elif type.startswith("bool"):
        type = torch.bool
    else:
        type = torch.float32
    return torch.zeros(shape, dtype=type, device=device)

register(id = "Trading-v0", entry_point = "__main__:EnvironmentWrapper")

class EnvironmentWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.actions = env.actions

        action_shape = transition_shape["actions"]["shape"]
        action_type = transition_shape["actions"]["type"]
        state_shape = transition_shape["states"]["shape"][1:]
        state_type = transition_shape["states"]["type"]
        print("action shape {}, action type {}, state shape {}, state type {}".format(action_shape, action_type,
                                                                                      state_shape, state_type))

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=state_shape,
            dtype=state_type,
        )

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info


def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk

class StateDataset():
    def __init__(self,
                 data_path: str = None,
                 assets_path: str = None,
                 fields_name: Dict[str, List[str]] = None,
                 states_path: str = None,
                 select_asset: str = None,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 timestamp_format: str = "%Y-%m-%d",
                 if_use_cs: bool = True,
                 if_use_ts: bool = True,
                 exp_path: str = None,
                 **kwargs
                 ):
        super(StateDataset, self).__init__()

        self.data_path = assemble_project_path(data_path)
        self.assets_path = assemble_project_path(assets_path)

        self.fields_name = fields_name

        self.prices_name = self.fields_name["prices"]

        self.states_path = assemble_project_path(states_path)
        assert os.path.exists(self.states_path), f"states_path: {self.states_path} not exists"

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.timestamp_format = timestamp_format
        self.if_use_cs = if_use_cs
        self.if_use_ts = if_use_ts

        self.assets = self._init_assets()
        self.select_asset = select_asset
        self.select_asset_index = self.assets.index(self.select_asset)

        self.assets_df = self._load_assets_df()
        self.features, self.prices = self._init_features()
        self.data_info = self._init_data_info()

    def _init_assets(self):

        assets = load_json(self.assets_path)
        if isinstance(assets, dict):
            asset_symbols = list(assets.keys())
        elif isinstance(assets, list):
            asset_symbols = [asset["symbol"] for asset in assets]
        else:
            raise ValueError("Unsupported assets format. Expected a dict or a list of dicts.")

        return asset_symbols

    def _load_assets_df(self):
        start_timestamp = pd.to_datetime(self.start_timestamp, format=self.timestamp_format) if self.start_timestamp else None
        end_timestamp = pd.to_datetime(self.end_timestamp, format=self.timestamp_format) if self.end_timestamp else None

        assets_df = {}
        for asset in self.assets:
            asset_path = os.path.join(self.data_path, "{}.csv".format(asset))
            asset_df = pd.read_csv(asset_path, index_col=0)
            asset_df.index = pd.to_datetime(asset_df.index)

            if start_timestamp and end_timestamp:
                asset_df = asset_df.loc[start_timestamp:end_timestamp]
            elif start_timestamp:
                asset_df = asset_df.loc[start_timestamp:]
            elif end_timestamp:
                asset_df = asset_df.loc[:end_timestamp]
            else:
                pass

            assets_df[asset] = asset_df
        return assets_df

    def _init_features(self):

        states = load_joblib(self.states_path)

        meta = states["meta"]
        items = states["items"]

        features = {}

        for timestamp, item in items.items():

            if self.if_use_cs and self.if_use_ts:

                ts_n_size = meta["ts_n_size"]

                factor_cs = item["factor_cs"]
                factor_ts = item["factor_ts"]

                embed_dim = factor_ts.shape[-1]

                # (n1, n2, n3, embed_dim)
                factor_ts = rearrange(factor_ts, "(n1 n2 n3) c -> n1 n2 n3 c", n1=ts_n_size[0], n2=ts_n_size[1], n3=ts_n_size[2], c = embed_dim)

                select_asset_factor_ts = factor_ts[:, self.select_asset_index, :, :]
                select_asset_factor_ts = rearrange(select_asset_factor_ts, "n1 n3 c -> (n1 n3) c", n1=ts_n_size[0], n3=ts_n_size[2], c = embed_dim)

                factors = np.concatenate([factor_cs, select_asset_factor_ts], axis=0)

                features[timestamp] = factors

            elif self.if_use_cs and not self.if_use_ts:

                factor = item["factor"]
                features[timestamp] = factor

            elif not self.if_use_cs and self.if_use_ts:

                n_size = meta["n_size"]
                factor = item["factor"]

                embed_dim = factor.shape[-1]

                # (n1, n2, n3, embed_dim)
                factor = rearrange(factor, "(n1 n2 n3) c -> n1 n2 n3 c", n1=n_size[0], n2=n_size[1], n3=n_size[2], c = embed_dim)

                select_asset_factor = factor[:, self.select_asset_index, :, :]
                select_asset_factor = rearrange(select_asset_factor, "n1 n3 c -> (n1 n3) c", n1=n_size[0], n3=n_size[2], c = embed_dim)

                features[timestamp] = select_asset_factor

        prices = self.assets_df[self.select_asset][self.prices_name]
        return features, prices

    def _init_data_info(self):
        data_info = {}
        count = 0

        first_asset = self.assets_df[self.assets[0]]
        for i in range(self.history_timestamps, len(first_asset) - self.future_timestamps):
            history_df = first_asset.iloc[i - self.history_timestamps: i]
            future_df = first_asset.iloc[i: i + self.future_timestamps]

            history_info = {
                "start_timestamp": history_df.index[0],
                "end_timestamp": history_df.index[-1],
                "start_index": i - self.history_timestamps,
                "end_index": i - 1,
            }

            future_info = {
                "start_timestamp": future_df.index[0],
                "end_timestamp": future_df.index[-1],
                "start_index": i,
                "end_index": i + self.future_timestamps - 1,
            }

            data_info[count] = {
                "history_info": history_info,
                "future_info": future_info,
            }

            count += 1

        return data_info

class Environment(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 initial_amount: float = 1e3,
                 transaction_cost_pct: float = 1e-3,
                 timestamp_format: str = "%Y-%m-%d",
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp="2008-04-01",
                 end_timestamp="2021-04-01",
                 **kwargs
                 ):
        super(Environment, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.assets = self.dataset.assets
        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format

        self.features, self.prices = self._init_features()
        self.data_info = self._init_data_info()
        index2timestamps = {key: value["history_info"]["end_timestamp"] for key, value in self.data_info.items()}
        self.index2timestamps = index2timestamps
        self.timestamp_min_index = min(index2timestamps.keys())
        self.timestamp_max_index = max(index2timestamps.keys())

        self.num_timestamps = len(self.data_info)

        self.hold_on_action = 1 # sell, hold, buy=>-1, 0, 1
        self.action_dim = 2 * self.hold_on_action + 1
        self.actions = ["SELL", "HOLD", "BUY"]

    def _init_features(self):
        features = self.dataset.features
        prices = self.dataset.prices
        prices = prices.loc[self.start_timestamp:self.end_timestamp]

        part_features = {}
        for timestamp in prices.index:
            timestamp = timestamp.strftime(self.timestamp_format)
            if timestamp in features:
                part_features[timestamp] = features[timestamp]

        return part_features, prices

    def _init_data_info(self):
        data_info = {}
        count = 0

        for i in range(self.history_timestamps, len(self.prices) - self.future_timestamps):
            history_df = self.prices.iloc[i - self.history_timestamps: i]
            future_df = self.prices.iloc[i: i + self.future_timestamps]

            history_info = {
                "start_timestamp": history_df.index[0],
                "end_timestamp": history_df.index[-1],
                "start_index": i - self.history_timestamps,
                "end_index": i - 1,
            }

            future_info = {
                "start_timestamp": future_df.index[0],
                "end_timestamp": future_df.index[-1],
                "start_index": i,
                "end_index": i + self.future_timestamps - 1,
            }

            data_info[count] = {
                "history_info": history_info,
                "future_info": future_info,
            }

            count += 1

        return data_info

    def init_timestamp_index(self):
        if self.mode == "train":
            timestamp = random.randint(0, 3 * (self.num_timestamps // 4))
        else:
            timestamp = 0
        return timestamp

    def get_timestamp(self, timestamp_index: int):
        return self.index2timestamps[timestamp_index]

    def current_value(self, price):
        return self.cash + self.position * price

    def get_price(self, timestamp_index: int):
        timestamp_datetime = self.get_timestamp(timestamp_index)
        prices = self.prices.loc[timestamp_datetime.strftime(self.timestamp_format)].values

        o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]
        price = adj

        return price

    def get_value(self,
                  cash: float,
                  postition: int,
                  price: float):
        value = cash + postition * price
        return value

    def reset(self, **kwargs):

        self.timestamp_index = self.init_timestamp_index()
        self.timestamp_datetime = self.get_timestamp(self.timestamp_index)
        self.next_timestamp_datetime = self.get_timestamp(self.timestamp_index + 1 if self.timestamp_index + 1 <= self.timestamp_max_index else self.timestamp_index)

        self.price, self.next_price = self.get_price(self.timestamp_index)

        state = self.features[self.timestamp_datetime.strftime(self.timestamp_format)]

        self.ret = 0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.total_return = 0
        self.total_profit = 0
        self.action = "HOLD"

        info= {
            "timestamp": self.timestamp_datetime.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "pre_value": self.pre_value,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": self.action
        }

        return state, info

    def eval_buy_position(self, price):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        # evaluate sell position
        return int(self.position)

    def buy(self, price, amount):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_buy_postion))

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

        if buy_position == 0:
            self.action = "HOLD"
        else:
            self.action = "BUY"

    def sell(self, price, amount):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.hold_on_action)) * eval_sell_postion))

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

        if sell_position == 0:
            self.action = "HOLD"
        else:
            self.action = "SELL"

    def noop(self, price, amount):
        self.value = self.current_value(price)

        self.action = "HOLD"

    def step(self, action: int = 0):

        action = action - self.hold_on_action

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.noop(self.price, amount=action)

        ret = (self.value - self.pre_value) / self.pre_value
        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100

        # next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.num_timestamps - 1:
            done = False
            truncted = False
        else:
            done = True
            truncted = True
        self.timestamp_datetime = self.get_timestamp(self.timestamp_index)
        self.state = self.features[self.timestamp_datetime.strftime(self.timestamp_format)] # next state
        self.price = self.get_price(self.timestamp_index) # next price

        reward = ret

        info = {
            "timestamp": self.timestamp_datetime.strftime(self.timestamp_format),
            "ret": self.ret,
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "discount": self.discount,
            "pre_value": self.pre_value,
            "value": self.value,
            "total_profit": self.total_profit,
            "total_return": self.total_return,
            "action": str(self.action)
        }

        # update pre_value
        self.pre_value = self.value

        return self.state, reward, done, truncted, info

class Actor(nn.Module):
    def __init__(self,
                 *args,
                 input_size = (64, 128),
                 embed_dim: int = 256,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 output_dim = 3,
                 **kwargs
                 ):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.output_dim = output_dim

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.encoder_layer = nn.Linear(
            self.input_size[1],
            embed_dim,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim,
                    hidden_features=embed_dim,
                    act_layer=nn.Tanh,
                    out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_layer = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

        self.proj = nn.Linear(
            self.input_size[0],
            self.output_dim,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):

        x = self.encoder_layer(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_layer(x).squeeze(-1)
        x = self.proj(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent).squeeze(-1)
        return pred

class Critic(nn.Module):
    def __init__(self,
                 *args,
                 input_size = (64, 128),
                 embed_dim: int = 256,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 output_dim = 3,
                 **kwargs
                 ):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.output_dim = output_dim

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.encoder_layer = nn.Linear(
            self.input_size[1],
            embed_dim,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim,
                    hidden_features=embed_dim,
                    act_layer=nn.Tanh,
                    out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_layer = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

        self.proj = nn.Linear(
            self.input_size[0],
            self.output_dim,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):

        x = self.encoder_layer(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_layer(x).squeeze(-1)
        x = self.proj(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent).squeeze(-1)
        return pred

class PPO(nn.Module):
    def __init__(self,
                 input_size=(64, 128),
                 embed_dim: int = 256,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = False,
                 action_dim=3,
                 **kwargs
                 ):

        super(PPO, self).__init__()

        self.actor = Actor(
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=action_dim,
        )

        self.critic = Critic(
            input_size=input_size,
            embed_dim=embed_dim,
            depth=depth,
            norm_layer=norm_layer,
            cls_embed=cls_embed,
            output_dim=1,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        if len(logits .shape) == 2:
            logits  = logits.unsqueeze(1)

        b, c, n = logits.shape

        logits = rearrange(logits , "b c n -> (b c) n", b=b, c=c, n=n)

        dis = Categorical(logits=logits)

        if action is None:
            action = dis.sample()

        probs = dis.log_prob(action)
        entropy = dis.entropy()
        value = self.critic(x)

        action = rearrange(action, "(b c) -> b c", b=b, c=c).squeeze(1)
        probs = rearrange(probs, "(b c) -> b c", b=b, c=c).squeeze(1)
        entropy = rearrange(entropy, "(b c) -> b c", b=b, c=c).squeeze(1)

        return action, probs, entropy, value

    def forward(self, *input, **kwargs):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "exp", "AAPL_day_dj30_dynamic_dual_vqvae.py"), help="config file path")

    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--tensorboard_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--if_remove", action="store_true", default=False)

    parser.add_argument("--tensorboard", action="store_true", default=True, help="enable tensorboard")
    parser.add_argument("--no_tensorboard", action="store_false", dest="tensorboard")
    parser.set_defaults(writer=True)

    parser.add_argument("--wandb", action="store_true", default=True, help="enable wandb")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb")
    parser.set_defaults(wandb=True)

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. build config
    config = build_config(assemble_project_path(args.config), args)

    # 2. set dtype
    dtype = to_torch_dtype(config.dtype)

    # 3. get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. init logger
    logger.init_logger(config.log_path, accelerator=None)
    if config.tensorboard:
        tensorboard_logger.init_logger(config.tensorboard_path, accelerator=None)
    if config.wandb:
        wandb_logger.init_logger(
            project=config.project,
            name=config.tag,
            config=config.to_dict(),
            dir=config.wandb_path,
            accelerator=None,
        )

    dataset_cfg = config.dataset
    dataset = StateDataset(
        **dataset_cfg
    )

    train_environment_cfg = config.train_environment
    train_environment_cfg.update({
        "dataset": dataset,
    })
    train_environment = Environment(
        **train_environment_cfg
    )
    valid_environment_cfg = config.valid_environment
    valid_environment_cfg.update({
        "dataset": dataset,
    })
    valid_environment = Environment(
        **valid_environment_cfg
    )
    test_environment_cfg = config.test_environment
    test_environment_cfg.update({
        "dataset": dataset,
    })
    test_environment = Environment(
        **test_environment_cfg
    )

    train_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env = deepcopy(train_environment),
                                     transition_shape = config.transition_shape, seed = config.seed + i)) for i in range(config.num_envs)
    ])

    valid_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(valid_environment),
                                                 transition_shape=config.transition_shape, seed=config.seed + i)) for i in range(1)
    ])

    test_environments = gym.vector.SyncVectorEnv([
        make_env("Trading-v0", env_params=dict(env=deepcopy(test_environment),
                                                 transition_shape=config.transition_shape, seed=config.seed + i)) for i in range(1)
    ])

    agent_cfg = config.agent
    agent = PPO(
        **agent_cfg
    ).to(device)

    policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, list(agent.actor.parameters())), lr=config.policy_learning_rate, eps=1e-5, weight_decay=0)
    value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(agent.critic.parameters())), lr=config.value_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    transition_shape = config.transition_shape
    obs = build_storage(shape = (config.num_steps, *transition_shape["states"]["shape"]),
                        type = transition_shape["states"]["type"], device = device)
    actions = build_storage(shape = (config.num_steps, *transition_shape["actions"]["shape"]),
                            type = transition_shape["actions"]["type"], device = device)
    logprobs = build_storage(shape = (config.num_steps, *transition_shape["logprobs"]["shape"]),
                                type = transition_shape["logprobs"]["type"], device = device)
    rewards = build_storage(shape = (config.num_steps, *transition_shape["rewards"]["shape"]),
                                type = transition_shape["rewards"]["type"], device = device)
    dones = build_storage(shape = (config.num_steps, *transition_shape["dones"]["shape"]),
                                type = transition_shape["dones"]["type"], device = device)
    values = build_storage(shape = (config.num_steps, *transition_shape["values"]["shape"]),
                                type = transition_shape["values"]["type"], device = device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()

    state, info = train_environments.reset()

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size
    num_critic_warm_up_updates = config.critic_warm_up_steps // config.batch_size

    is_warmup = True
    prefix = "train"
    for update in range(1, num_updates + 1 + num_critic_warm_up_updates):
        if is_warmup and update > num_critic_warm_up_updates:
            is_warmup = False

        # Annealing the rate if instructed to do so.
        if config.anneal_lr and not is_warmup:
            frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * config.policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * config.value_learning_rate

        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncted, info = train_environments.step(action.cpu().numpy())
            # print(info)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                print("final_info", info["final_info"])
                for info_item in info["final_info"]:
                    if info_item is not None:
                        logger.info(f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                        tensorboard_logger.add_scalar(f"{prefix}/total_return", info_item["total_return"], global_step)
                        tensorboard_logger.add_scalar(f"{prefix}/total_profit", info_item["total_profit"], global_step)

                        wandb_dict = {
                            f"{prefix}/total_return": info_item["total_return"],
                            f"{prefix}/total_profit": info_item["total_profit"],
                        }
                        wandb_logger.log(wandb_dict)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + transition_shape["states"]["shape"][1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.view(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)

        for epoch in range(config.update_epochs):
            if kl_explode:
                break
            # update value
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.value_minibatch_size):
                end = start + config.value_minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = agent.get_value(b_obs[mb_inds])

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * config.vf_coef

                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                value_optimizer.step()

            if is_warmup:
                continue

            policy_optimizer.zero_grad()
            # update policy
            for start in range(0, config.batch_size, config.policy_minibatch_size):
                if policy_update_steps % config.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + config.policy_minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / config.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss
                loss /= config.gradient_checkpointing_steps

                loss.backward()

                if policy_update_steps % config.gradient_checkpointing_steps == 0:
                    if config.target_kl is not None:
                        if total_approx_kl > config.target_kl:
                            policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= config.gradient_checkpointing_steps
                            # print("break", policy_update_steps)
                            break

                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    policy_optimizer.step()
                    policy_optimizer.zero_grad()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        tensorboard_logger.add_scalar(f"{prefix}/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        tensorboard_logger.add_scalar(f"{prefix}/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        tensorboard_logger.add_scalar(f"{prefix}/value_loss", v_loss.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/policy_loss", pg_loss.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/entropy", entropy_loss.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/old_approx_kl", old_approx_kl.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/approx_kl", approx_kl.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/total_approx_kl", total_approx_kl.item(), global_step)
        tensorboard_logger.add_scalar(f"{prefix}/policy_update_times", policy_update_steps // config.gradient_checkpointing_steps, global_step)
        tensorboard_logger.add_scalar(f"{prefix}/clipfrac", num_clipfracs, global_step)
        tensorboard_logger.add_scalar(f"{prefix}/explained_variance", explained_var, global_step)
        tensorboard_logger.add_scalar(f"{prefix}/SPS", global_step / (time.time() - start_time), global_step)

        wandb_dict = {
            f"{prefix}/policy_learning_rate": policy_optimizer.param_groups[0]["lr"],
            f"{prefix}/value_learning_rate": value_optimizer.param_groups[0]["lr"],
            f"{prefix}/value_loss": v_loss.item(),
            f"{prefix}/policy_loss": pg_loss.item(),
            f"{prefix}/entropy": entropy_loss.item(),
            f"{prefix}/old_approx_kl": old_approx_kl.item(),
            f"{prefix}/approx_kl": approx_kl.item(),
            f"{prefix}/total_approx_kl": total_approx_kl.item(),
            f"{prefix}/policy_update_times": policy_update_steps // config.gradient_checkpointing_steps,
            f"{prefix}/clipfrac": num_clipfracs,
            f"{prefix}/explained_variance": explained_var,
            f"{prefix}/SPS": global_step / (time.time() - start_time),
        }
        wandb_logger.log(wandb_dict)

        logger.info(f"SPS: {global_step}, {(time.time() - start_time)}")

        if global_step // config.check_steps != pre_global_step // config.check_steps:
            validate_agent(config, agent, valid_environments, logger, tensorboard_logger, wandb_logger, device, global_step, config.exp_path)
            torch.save(agent.state_dict(), os.path.join(config.checkpoint_path, "{}.pth".format(global_step // config.check_steps)))
        pre_global_step = global_step

    validate_agent(config, agent, valid_environments, logger, tensorboard_logger, wandb_logger, device, global_step, config.exp_path)
    torch.save(agent.state_dict(), os.path.join(config.checkpoint_path, "{}.pth".format(global_step // config.check_steps + 1)))

    train_environments.close()
    valid_environments.close()
    test_environments.close()
    tensorboard_logger.close()
    wandb_logger.finish()

def validate_agent(config, agent, envs, logger, writer, wandb,  device, global_step, exp_path):

    prefix = "valid"

    rets = []
    trading_records = {
        "timestamp": [],
        "value": [],
        "cash": [],
        "position": [],
        "ret": [],
        "price": [],
        "discount": [],
        "total_profit": [],
        "total_return": [],
        "action": [],
    }

    # TRY NOT TO MODIFY: start the game
    state, info = envs.reset()
    rets.append(info["ret"])

    next_obs = torch.Tensor(state).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    while True:
        obs = next_obs
        dones = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            logits = agent.actor(next_obs)
            action = torch.argmax(logits, dim=1)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, truncted, info = envs.step(action.cpu().numpy())
        rets.append(info["ret"])
        trading_records["timestamp"].append(info["timestamp"])
        trading_records["value"].append(info["value"])
        trading_records["cash"].append(info["cash"])
        trading_records["position"].append(info["position"])
        trading_records["ret"].append(info["ret"])
        trading_records["price"].append(info["price"])
        trading_records["discount"].append(info["discount"])
        trading_records["total_profit"].append(info["total_profit"])
        trading_records["total_return"].append(info["total_return"])
        # trading_records["action"].append(envs.actions[action.cpu().numpy()])
        trading_records["action"].append(action.cpu().numpy())

        if trading_records["action"][-1] != info["action"]:
            trading_records["action"][-1] = info["action"]

        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        if "final_info" in info:
            print("val final_info", info["final_info"])
            for info_item in info["final_info"]:
                if info_item is not None:
                    logger.info(f"global_step={global_step}, total_return={info_item['total_return']}, total_profit = {info_item['total_profit']}")
                    writer.add_scalar(f"{prefix}/total_return", info_item["total_return"], global_step)
                    writer.add_scalar(f"{prefix}/total_profit", info_item["total_profit"], global_step)

                    wandb_dict = {
                        f"{prefix}/total_return": info_item["total_return"],
                        f"{prefix}/total_profit": info_item["total_profit"],
                    }

                    wandb.log(wandb_dict)

            break

    rets = np.array(rets)
    arr = ARR(rets)       # take as reward
    sr = SR(rets)
    dd = MDD(rets)
    mdd = MDD(rets)
    cr = CR(rets, mdd=mdd)
    sor = SOR(rets, dd=dd)
    vol = VOL(rets)

    writer.add_scalar(f"{prefix}/ARR%", arr * 100, global_step)
    writer.add_scalar(f"{prefix}/SR", sr, global_step)
    writer.add_scalar(f"{prefix}/CR", cr, global_step)
    writer.add_scalar(f"{prefix}/SOR", sor, global_step)
    writer.add_scalar(f"{prefix}/DD", dd, global_step)
    writer.add_scalar(f"{prefix}/MDD%", mdd * 100, global_step)
    writer.add_scalar(f"{prefix}/VOL", vol, global_step)

    wandb_dict = {
        f"{prefix}/ARR%": arr * 100,
        f"{prefix}/SR": sr,
        f"{prefix}/CR": cr,
        f"{prefix}/SOR": sor,
        f"{prefix}/DD": dd,
        f"{prefix}/MDD%": mdd * 100,
        f"{prefix}/VOL": vol,
    }
    wandb.log(wandb_dict)

    logger.info(
        f"global_step={global_step}, ARR%={arr * 100}, SR={sr}, CR={cr}, SOR={sor}, DD={dd}, MDD%={mdd * 100}, VOL={vol}"
    )

    # print(f"trading_records is   {trading_records}")
    for key in trading_records.keys():
        trading_records[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in trading_records[key]]

    save_json(trading_records, os.path.join(exp_path, "valid_records.json"))

if __name__ == '__main__':
    main()