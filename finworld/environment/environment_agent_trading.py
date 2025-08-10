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
from finworld.utils import assemble_project_path, TradingRecords, truncate_content
from finworld.utils import get_token_count
from finworld.utils import get_start_end_timestamp


__all__ = ['EnvironmentAgentTrading']

@dataclass
class Record():
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
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
            values = row[1]

            content = values["summary"] if "summary" in values else values["content"]
            content = truncate_content(content, max_length=1024)

            if content is not None:
                content = content.replace('\n', '')
            title = row[1]['title']
            news_string += f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {title} | {content}\n"

    record_string = record.to_markdown(index=False)
    note_string  = "1. `timestamp`: the timestamp of the record\n"
    note_string += "2. `open`: Open price\n"
    note_string += "3. `high`: High price\n"
    note_string += "4. `low`: Low price\n"
    note_string += "5. `close`: Close price\n"
    note_string += "6. `volume`: Volume of the asset traded\n"
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
class EnvironmentAgentTrading(gym.Env):
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
        record_max_len: int = 32,
        valid_action_max_len: int = 8,
        single_text_max_tokens: int = 1024,
        single_text_min_tokens: int = 256,
        daily_sample_texts: int = 2,
        **kwargs,
    ):
        super(EnvironmentAgentTrading, self).__init__()

        self.mode = mode
        self.dataset = dataset
        self.symbol = self.dataset.symbol
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        asset_info = self.dataset.asset_info

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
        self.record_max_len = record_max_len
        self.valid_action_max_len = valid_action_max_len
        self.single_text_max_tokens = single_text_max_tokens
        self.single_text_min_tokens = single_text_min_tokens
        self.daily_sample_texts = daily_sample_texts

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_df = self.res_info['features_df']
        self.original_prices_df = self.res_info['original_prices_df']
        self.news_df = self.res_info['news_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

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

    def get_price_full_df(self):
        start_timestamp_index = self.timestamp_min_index
        end_timestamp_index = self.timestamp_max_index

        start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"]
        end_timestamp = self.timestamp_info[end_timestamp_index]["end_timestamp"]

        original_prices_df = self._get_dataitem(self.original_prices_df,
                                       start_timestamp,
                                       end_timestamp)
        return original_prices_df

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
        news = self._get_dataitem(self.news_df, start_timestamp, end_timestamp)

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
        prompt += f"Today is {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}, and the current price, cash, and position are {self.price:.2f}, {self.cash:.2f}, and {self.position:04d}.\n"
        prompt_token_nums = get_token_count(prompt)

        state = dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            prompt=prompt,
            prompt_token_nums=prompt_token_nums,
        )
        state.update(self.asset_info)

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

    def _init_record(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)

        rows = list(price.iterrows())

        for row in rows[:-1]:
            timestamp = row[0]
            timestamp_string = timestamp.strftime(self.level_format.value)
            close, high, low, open, volume = row[1].values
            self.last_record = Record(
                timestamp=timestamp_string,
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
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
        timestamp_string = timestamp.strftime(self.level_format.value)
        close, high, low, open, volume = rows[-1][1].values
        self.last_record = Record(
            timestamp=timestamp_string,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
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

        # init record
        self._init_record()

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

        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        # next record
        close, high, low, open, volume = self.get_price_full(timestamp_index=self.timestamp_index)
        self.last_record = Record(
            timestamp=self.timestamp_string,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
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
    history_timestamps = 5
    future_timestamps = 0
    start_timestamp = "2015-05-01"
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
        if_norm=False,
        if_use_future=False,
        if_use_temporal=True,
        if_norm_temporal=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
    )

    env_cfg: dict[str, Any] = dict(
        type='EnvironmentAgentTrading',
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
        record_max_len=32,
        valid_action_max_len=8,
        single_text_max_tokens=1024,
        single_text_min_tokens=256,
        daily_sample_texts=2,
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

    for step in range(500):
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, truncted, info = environment.step(action)

        if step == 10:
            print(next_state['prompt'])
            exit()

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

        if "final_info" in info:
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