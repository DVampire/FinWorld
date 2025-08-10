import os
import warnings
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

from typing import Any, Optional
import random
from mathruler.grader import extract_boxed_content

import gym
import numpy as np
import pandas as pd

from finworld.registry import DATASET, ENVIRONMENT
from finworld.utils import assemble_project_path
from finworld.utils import get_token_count



__all__ = ['EnvironmentLLMTrading']

@dataclass
class Record():
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    price: float
    cash: float
    position: int
    pre_value: Optional[float]
    action: Optional[str]
    post_value: Optional[float]
    ret: Optional[float]


def sample_news(df: pd.DataFrame, sample_texts: int = 2):
    """
    Sample news from the news_df.
    :param news_df: DataFrame of news
    :param sample_texts: number of texts to sample
    :return: sampled news
    """
    if len(df) == 0:
        return None
    else:
        df = df.reset_index(drop=False)
        df['date'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df = df.groupby('date').apply(lambda x: x.sample(n=min(sample_texts, len(x)), random_state=0)).reset_index(drop=True)
        df.drop(columns=['date'], inplace=True)
        df.set_index('timestamp', inplace=True)
        return df

def convert_dataframe_to_markdown(
        price: pd.DataFrame,
        news: pd.DataFrame,
        record: pd.DataFrame,
        valid_action: pd.DataFrame,
    ):

    price_string = price.to_markdown(index=False)

    news_string = f"**Timestamp | Title | Content**\n"
    if news is None:
        news_string += f"**No news available**\n"
    else:
        for row in news.iterrows():
            timestamp = row[0]
            content = row[1]['content']
            if content is not None:
                content = content.replace('\n', '')
            title = row[1]['title']
            news_string += f"{timestamp.strftime('%Y-%m-%d')} | {title} | {content} \n"

    record_string = record.to_markdown(index=False)
    note_string  = "1. `timestamp`: the timestamp of the record\n"
    note_string += "2. `open`: Open price\n"
    note_string += "3. `high`: High price\n"
    note_string += "4. `low`: Low price\n"
    note_string += "5. `close`: Close price\n"
    note_string += "6. `adj_close`: Adjusted close price\n"
    note_string += "7. `price`: Current price (adj_close price)\n"
    note_string += "8. `cash`: Current cash\n"
    note_string += "9. `position`: Current position\n"
    note_string += "10. `pre_value`: Previous total value, `value = cash + position * price`\n"
    note_string += "11. `action`: Action taken, `BUY`, `SELL`, or `HOLD`\n"
    note_string += "12. `post_value`: Current total value\n"
    note_string += "13. `ret`: Return, `ret = (post_value - pre_value) / pre_value`\n"

    valid_action_string = valid_action.to_markdown(index=False)

    res_strings = dict(
        price=price_string,
        news=news_string,
        record=record_string,
        note=note_string,
        valid_action=valid_action_string,
    )

    return res_strings


@ENVIRONMENT.register_module(force=True)
class EnvironmentLLMTrading(gym.Env):
    def __init__(
        self,
        *args,
        mode: str = "train",
        dataset: Any = None,
        select_asset: str = '',
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        timestamp_format: str = '%Y-%m-%d',
        history_timestamps: int = 32,
        step_timestamps: int = 1,
        future_timestamps: int = 32,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        gamma: float = 0.99,
        record_max_len: int = 32,
        valid_action_max_len: int = 8,
        single_text_max_tokens: int = 1024,
        single_text_min_tokens: int = 256,
        daily_sample_texts: int = 2,
        **kwargs,
    ):
        super(EnvironmentLLMTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.select_asset = select_asset
        assert self.select_asset in self.dataset.symbols and self.select_asset is not None, \
            f"select_asset {self.select_asset} not in assets {self.dataset.symbols}"

        asset_info = self.dataset.assets_info[self.select_asset]

        self.asset_info = dict(
            asset_symbol=asset_info['symbol'],
            asset_name=asset_info['companyName'],
            asset_exchange=asset_info['exchange'],
            asset_sector=asset_info['sector'],
            asset_industry=asset_info['industry'],
            asset_description=asset_info['description'],
        )

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.prices_name = self.dataset.prices_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.timestamp_format = timestamp_format
        self.gamma = gamma

        self.single_text_max_tokens = single_text_max_tokens
        self.single_text_min_tokens = single_text_min_tokens
        self.daily_sample_texts = daily_sample_texts

        res_info = self._init_features()
        self.timestamp_info = res_info['timestamp_info']
        self.features_df = res_info['features_df']
        self.prices_df = res_info['prices_df']
        self.news_df = res_info['news_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

        self.record_max_len = record_max_len
        self.valid_action_max_len = valid_action_max_len
        self.record_df = pd.DataFrame() # record the trading history
        self.valid_action_df = pd.DataFrame() # record the valid action

    def _filter_news(self, df: pd.DataFrame, text_name: str):
        assert text_name in df.columns, f"news_df must have {text_name} column"

        df['token_count'] = df[text_name].apply(lambda x: get_token_count(x))
        df = df[df['token_count'] >= self.single_text_min_tokens]
        df = df[df['token_count'] <= self.single_text_max_tokens]
        df.drop(columns=['token_count'], inplace=True)

        return df

    def _init_features(self):
        timestamp_info = {}
        multi_asset_meta_info = self.dataset.multi_asset_meta_info.items
        for key, value in multi_asset_meta_info.items():
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
        assert (
            self.num_timestamps == len(timestamp_info)
        ), f'num_timestamps {self.num_timestamps} != len(data_info) {len(timestamp_info)}'

        features = self.dataset.asset_data[self.select_asset].features
        prices = self.dataset.asset_data[self.select_asset].original_prices
        news = self.dataset.asset_data[self.select_asset].news_df

        features_df = features
        prices_df = prices
        news_df = self._filter_news(df=news, text_name='content')

        res_info = dict(
            timestamp_info=timestamp_info,
            features_df=features_df,
            prices_df=prices_df,
            news_df=news_df,
        )

        return res_info

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

        return float(price)

    def get_price_full(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        end_timestamp = timestamp_info['end_timestamp']

        prices = self.prices_df.loc[end_timestamp].values
        o, h, l, c, adj = prices[0], prices[1], prices[2], prices[3], prices[4]

        return float(o), float(h), float(l), float(c), float(adj)

    def get_state(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self.prices_df.loc[start_timestamp:end_timestamp]
        news = self.news_df.loc[start_timestamp:end_timestamp]
        record = self.record_df
        valid_action = self.valid_action_df

        sampled_news = sample_news(df=news, sample_texts=self.daily_sample_texts)

        # convert to markdown
        strings = convert_dataframe_to_markdown(
            price=price,
            news=sampled_news,
            record=record,
            valid_action=valid_action,
        )
        price_string = strings['price']
        news_string = strings['news']
        record_string = strings['record']
        note_string = strings['note']
        valid_action_string = strings['valid_action']

        prompt = f"# Name: {self.asset_info['asset_name']}, Symbol: ({self.asset_info['asset_symbol']})\n"
        prompt += f"## Price\n{price_string}\n"
        prompt += f"## News\n{news_string}\n"
        prompt += f"## Record\n{record_string}\n"
        prompt += f"## History Valid Action\n{valid_action_string}\n"
        prompt += f"## Note\n{note_string}\n"
        prompt += f"Today is {end_timestamp.strftime(self.timestamp_format)}, and the current price, cash, and position are {self.price:.2f}, {self.cash:.2f}, and {self.position:04d}.\n"
        prompt += f"Please conduct an in-depth analysis of the above information, systematically identify and evaluate all key factors relevant to the trading decision, and provide a clear final answer to BUY, SELL, or HOLD based on the analysis.\n"
        prompt += "The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.\nExample: \\boxed{BUY}.\n"

        prompt_token_nums = get_token_count(prompt)

        state = dict(
            timestamp=end_timestamp.strftime(self.timestamp_format),
            prompt=prompt,
            prompt_token_nums=prompt_token_nums,
        )
        state.update(self.asset_info)

        return state

    def eval_buy_position(self, cash: float, price: float):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self, position: int):
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

    def _init_record(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self.prices_df.loc[start_timestamp:end_timestamp]

        rows = list(price.iterrows())

        for row in rows[:-1]:
            timestamp = row[0]
            o, h, l, c, adj = row[1].values
            self.last_record = Record(
                timestamp=timestamp.strftime(self.timestamp_format),
                open=o,
                high=h,
                low=l,
                close=c,
                adj_close=adj,
                price=self.price,
                cash=self.cash,
                position=self.position,
                pre_value=self.initial_amount,
                action='HOLD',
                post_value=self.initial_amount,
                ret=0.0,
            )
            self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(self.last_record)])], ignore_index=True)

        # last record, because the action is not predicted, so the pre_value, action, post_value, ret are None
        timestamp = rows[-1][0]
        o, h, l, c, adj = rows[-1][1].values
        self.last_record = Record(
            timestamp=timestamp.strftime(self.timestamp_format),
            open=o,
            high=h,
            low=l,
            close=c,
            adj_close=adj,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=None,
            action=None,
            post_value=None,
            ret=None,
        )
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(self.last_record)])], ignore_index=True)

        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']

    def _add_record(self, record):
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(record)])], ignore_index=True)

        record_max_len = min(self.record_max_len, len(self.record_df))
        self.record_df = self.record_df[-record_max_len:]
        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']

        valid_action_max_len = min(self.valid_action_max_len, len(self.valid_action_df))
        self.valid_action_df = self.valid_action_df[-valid_action_max_len:]

    def _update_record(self, record):

        last_record = self.record_df.iloc[-1]

        last_record['pre_value'] = record.pre_value
        last_record['action'] = record.action
        last_record['post_value'] = record.post_value
        last_record['ret'] = record.ret

        self.record_df.iloc[-1] = last_record

        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']
        valid_action_max_len = min(self.valid_action_max_len, len(self.valid_action_df))
        self.valid_action_df = self.valid_action_df[-valid_action_max_len:]

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)
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

        # init record
        self._init_record()

        # after init record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp.strftime(self.timestamp_format),
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

        if isinstance(action, str):
            action = self._extract_action(action)
        elif isinstance(action, np.ndarray):
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

        self.cash = res_info['cash']
        self.position = res_info['position']
        self.value = res_info['value']
        self.action = res_info['action']
        self.action_label = res_info['action_label']

        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)

        # update record
        self.last_record.pre_value = self.pre_value
        self.last_record.action = self.action_label
        self.last_record.post_value = self.value
        self.last_record.ret = ret
        self._update_record(self.last_record)

        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100
        reward = ret

        # next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.timestamp = self.get_timestamp(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        # next record
        o, h, l, c, adj = self.get_price_full(timestamp_index=self.timestamp_index)
        self.last_record = Record(
            timestamp=self.timestamp.strftime(self.timestamp_format),
            open=o,
            high=h,
            low=l,
            close=c,
            adj_close=adj,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=None,
            action=None,
            post_value=None,
            ret=None,
        )
        self._add_record(self.last_record)

        # after update record, get the state
        self.state = self.get_state(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp.strftime(self.timestamp_format),
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
    select_asset = 'AAPL'

    dataset = dict(
        type='MultiAssetDataset',
        data_path='workdir/processed_dj30',
        assets_path='configs/_asset_list_/dj30.json',
        fields_name={
            'features': [
                'open',
                'high',
                'low',
                'close',
                'adj_close',
                'kmid',
                'kmid2',
                'klen',
                'kup',
                'kup2',
                'klow',
                'klow2',
                'ksft',
                'ksft2',
                'roc_5',
                'roc_10',
                'roc_20',
                'roc_30',
                'roc_60',
                'ma_5',
                'ma_10',
                'ma_20',
                'ma_30',
                'ma_60',
                'std_5',
                'std_10',
                'std_20',
                'std_30',
                'std_60',
                'beta_5',
                'beta_10',
                'beta_20',
                'beta_30',
                'beta_60',
                'max_5',
                'max_10',
                'max_20',
                'max_30',
                'max_60',
                'min_5',
                'min_10',
                'min_20',
                'min_30',
                'min_60',
                'qtlu_5',
                'qtlu_10',
                'qtlu_20',
                'qtlu_30',
                'qtlu_60',
                'qtld_5',
                'qtld_10',
                'qtld_20',
                'qtld_30',
                'qtld_60',
                'rank_5',
                'rank_10',
                'rank_20',
                'rank_30',
                'rank_60',
                'imax_5',
                'imax_10',
                'imax_20',
                'imax_30',
                'imax_60',
                'imin_5',
                'imin_10',
                'imin_20',
                'imin_30',
                'imin_60',
                'imxd_5',
                'imxd_10',
                'imxd_20',
                'imxd_30',
                'imxd_60',
                'rsv_5',
                'rsv_10',
                'rsv_20',
                'rsv_30',
                'rsv_60',
                'cntp_5',
                'cntp_10',
                'cntp_20',
                'cntp_30',
                'cntp_60',
                'cntn_5',
                'cntn_10',
                'cntn_20',
                'cntn_30',
                'cntn_60',
                'cntd_5',
                'cntd_10',
                'cntd_20',
                'cntd_30',
                'cntd_60',
                'corr_5',
                'corr_10',
                'corr_20',
                'corr_30',
                'corr_60',
                'cord_5',
                'cord_10',
                'cord_20',
                'cord_30',
                'cord_60',
                'sump_5',
                'sump_10',
                'sump_20',
                'sump_30',
                'sump_60',
                'sumn_5',
                'sumn_10',
                'sumn_20',
                'sumn_30',
                'sumn_60',
                'sumd_5',
                'sumd_10',
                'sumd_20',
                'sumd_30',
                'sumd_60',
                'vma_5',
                'vma_10',
                'vma_20',
                'vma_30',
                'vma_60',
                'vstd_5',
                'vstd_10',
                'vstd_20',
                'vstd_30',
                'vstd_60',
                'wvma_5',
                'wvma_10',
                'wvma_20',
                'wvma_30',
                'wvma_60',
                'vsump_5',
                'vsump_10',
                'vsump_20',
                'vsump_30',
                'vsump_60',
                'vsumn_5',
                'vsumn_10',
                'vsumn_20',
                'vsumn_30',
                'vsumn_60',
                'vsumd_5',
                'vsumd_10',
                'vsumd_20',
                'vsumd_30',
                'vsumd_60',
            ],
            'prices': [
                'open',
                'high',
                'low',
                'close',
                'adj_close',
            ],
            'temporals': [
                'day',
                'weekday',
                'month',
            ],
            'labels': ['ret1', 'mov1'],
        },
        if_norm=True,
        if_norm_temporal=False,
        if_use_future=False,
        if_use_text=True,
        scaler_cfg=dict(
            type='WindowedScaler',
            window_size=5,
        ),
        scaler_file='scalers.joblib',
        scaled_data_file='scaled_data.joblib',
        history_timestamps = 5,
        future_timestamps = 5,
        start_timestamp="2024-05-01",
        end_timestamp="2025-01-01",
        timestamp_format='%Y-%m-%d',
        exp_path=assemble_project_path(os.path.join('workdir', 'test')),
    )

    env_cfg: dict[str, Any] = dict(
        type='EnvironmentAgentTrading',
        dataset=None,
        select_asset=select_asset,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        timestamp_format='%Y-%m-%d',
        start_timestamp="2024-05-01",
        end_timestamp="2025-01-01",
        history_timestamps = 5,
        future_timestamps = 5,
        step_timestamps=1,
        record_max_len = 5,
        valid_action_max_len = 5,
    )

    dataset = DATASET.build(dataset)

    env_cfg.update(
        dict(
            dataset=dataset,
        )
    )

    environment = ENVIRONMENT.build(env_cfg)

    state, info = environment.reset()
    print(state['prompt_token_nums'])

    for step in range(20):
        action = np.random.choice([0, 1, 2])
        state, reward, done, truncted, info = environment.step(action)
        print(state['prompt_token_nums'])
        print(
            f'step: {step}, action: {action}, reward: {reward}, done: {done}, truncted: {truncted}, info: {info}'
        )