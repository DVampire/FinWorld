import os
import json
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Union
from datasets import load_dataset
from pandas import DataFrame
from copy import deepcopy

from finworld.utils import get_tag_name
from finworld.registry import DATASET
from finworld.registry import SCALER
from finworld.utils import assemble_project_path
from finworld.data.collate_fn import SingleAssetPriceTextCollateFn
from finworld.utils import get_start_end_timestamp, TimeLevel, TimeLevelFormat

@DATASET.register_module(force=True)
class SingleAssetDataset(Dataset):

    def __init__(self,
                 symbol: str = None,
                 data_path: str = None,
                 enabled_data_configs: List[Dict[str, Any]] = None,
                 if_norm: bool = True,
                 if_use_temporal: bool = True,
                 if_use_future: bool = True,
                 if_norm_temporal: bool = True,
                 scaler_cfg: Dict[str, Any] = None,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 start_timestamp: str = None,
                 end_timestamp: str = None,
                 level="1day",
                 **kwargs
                 ):
        super(Dataset, self).__init__()

        self.symbol = symbol
        self.data_path = assemble_project_path(data_path)
        self.enabled_data_configs = enabled_data_configs
        self.if_norm = if_norm
        self.if_use_future = if_use_future
        self.if_use_temporal = if_use_temporal
        self.if_norm_temporal = if_norm_temporal
        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.scaler_cfg = scaler_cfg
        self.level = TimeLevel.from_string(level)
        self.level_format = TimeLevelFormat.from_string(level)

        if self.if_use_future:
            assert future_timestamps > 0, "If if_use_future is True, future_timestamps must be greater than 0."

        # load info
        self.asset_info_from_file = self._load_asset_info()
        self.enabled_data_info = self._load_enabled_data_info()
        self.if_use_features = self.enabled_data_info.get("if_use_features", False)
        self.if_use_news = self.enabled_data_info.get("if_use_news", False)

        # process data
        self.asset_data = self._init_asset_data()
        self.asset_info = self.asset_info_from_file
        self.asset_meta_info = self._init_asset_meta_info()

    def _load_asset_info(self):
        """
        Load assets from the assets file.
        :return:
        """
        meta_info_path = os.path.join(self.data_path, "meta_info.json")

        with open(meta_info_path) as f:
            meta_info = json.load(f)

        symbols_info = meta_info["symbols_info"]
        assert self.symbol in symbols_info, f"Asset {self.symbol} not found in meta_info.json"
        asset_info = symbols_info[self.symbol]

        return asset_info

    def _load_enabled_data_info(self):
        """
        Load data info from enabled_data_configs.
        Returns:
        {
            "prices": [{
                "asset_name": "asset1",
                "source": "source1",
                "data_type": "price",
                "level": "1day",
                "tag": "asset1_source1_price_1day"
            }, ...],
            "features": [{
                "asset_name": "asset1",
                "source": "source1",
                "data_type": "feature",
                "level": "1day",
                "tag": "asset1_source1_feature_1day"
            }, ...],
            "news": [{
                "asset_name": "asset1",
                "source": "source1",
                "data_type": "news",
                "level": "1day",
                "tag": "asset1_source1_news_1day"
            }, ...]
            "if_use_features": bool,
            "if_use_news": bool,
        }
        """

        meta_info_path = os.path.join(self.data_path, "meta_info.json")

        with open(meta_info_path) as f:
            meta_info = json.load(f)

        tags = meta_info["tags"]

        enabled_data = []
        for item in self.enabled_data_configs:
            asset_name = item.get("asset_name")
            source = item.get("source")
            data_type = item.get("data_type")
            level = item.get("level")

            tag = get_tag_name(
                assets_name=asset_name,
                source=source,
                data_type=data_type,
                level=level
            )
            assert tag in tags, f"Tag {tag} not found in meta_info.json"

            enabled_data.append(dict(
                asset_name=asset_name,
                source=source,
                data_type=data_type,
                level=level,
                tag=tag
            ))

        # check data info
        prices_data_info = [item for item in enabled_data if item["data_type"] == "price"]
        features_data_info = [item for item in enabled_data if item["data_type"] == "feature"]
        news_data_info = [item for item in enabled_data if item["data_type"] == "news"]
        assert len(prices_data_info) == 1, "Price data must be exactly one."
        assert len(features_data_info) <= 1, "Only one feature data is allowed."
        assert len(prices_data_info) <= 1, "Only one price data is allowed."

        if_use_features = len(features_data_info) > 0
        if_use_news = len(news_data_info) > 0

        enabled_data_info = dict(
            prices_data_info=prices_data_info,
            features_data_info=features_data_info,
            news_data_info=news_data_info,
            if_use_features=if_use_features,
            if_use_news=if_use_news,
        )

        return enabled_data_info

    def _load_dataframe(self,
                        enabled_info: Union[Dict[str, Any], List[Dict[str, Any]]],
                        symbol: str = None
                        ):

        start_timestamp = pd.to_datetime(self.start_timestamp) \
            if self.start_timestamp else None
        end_timestamp = pd.to_datetime(self.end_timestamp) \
            if self.end_timestamp else None

        if isinstance(enabled_info, Dict):
            tag = enabled_info["tag"]
            data_files = os.path.join(self.data_path, tag, "{}.jsonl".format(symbol))
            asset_df = load_dataset(
                "json",
                data_files=data_files,
            )['train'].to_pandas()
        elif isinstance(enabled_info, List):
            data_files = [os.path.join(self.data_path, item["tag"],
                                       "{}.jsonl".format(symbol)) for item in enabled_info]
            asset_df = load_dataset(
                "json",
                data_files=data_files,
            )['train'].to_pandas()
        else:
            raise ValueError("Enabled_info must be a Dict or List of Dicts.")

        asset_df["timestamp"] = pd.to_datetime(asset_df["timestamp"])
        asset_df = asset_df.drop_duplicates(subset=["timestamp"], keep="first")
        asset_df = asset_df.sort_values(by="timestamp")
        asset_df.set_index("timestamp", inplace=True)

        if start_timestamp and end_timestamp:
            asset_df = asset_df[(asset_df.index >= start_timestamp) & (asset_df.index <= end_timestamp)]
        elif start_timestamp:
            asset_df = asset_df[(asset_df.index >= start_timestamp)]
        elif end_timestamp:
            asset_df = asset_df.loc[(asset_df.index <= end_timestamp)]
        else:
            raise ValueError("At least one of start_timestamp or end_timestamp must be provided.")

        return asset_df

    def _cal_time(self,
                  df: DataFrame,
                  windows: List[int] = None,
                  level: TimeLevel = TimeLevel.DAY):
        """
        Time factor.
        """
        df = deepcopy(df)
        df['timestamp'] = df.index
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df["year"] = df['timestamp'].dt.year - df['timestamp'].dt.year.min()
        df["month"] = df['timestamp'].dt.month
        df["weekday"] = df['timestamp'].dt.weekday
        df["day"] = df['timestamp'].dt.day
        factors_info = {
            'year': 'Year extracted from timestamp',
            'month': 'Month extracted from timestamp',
            'weekday': 'Weekday extracted from timestamp (0=Monday, 6=Sunday)',
            'day': 'Day of the month extracted from timestamp'
        }

        if level == TimeLevel.HOUR:
            df["hour"] = df['timestamp'].dt.hour
            factors_info['hour'] = 'Hour extracted from timestamp'
        elif level == TimeLevel.MINUTE:
            df["hour"] = df['timestamp'].dt.hour
            df["minute"] = df['timestamp'].dt.minute
            factors_info['hour'] = 'Hour extracted from timestamp'
            factors_info['minute'] = 'Minute extracted from timestamp'
        elif level == TimeLevel.SECOND:
            df["hour"] = df['timestamp'].dt.hour
            df["minute"] = df['timestamp'].dt.minute
            df["second"] = df['timestamp'].dt.second
            factors_info['hour'] = 'Hour extracted from timestamp'
            factors_info['minute'] = 'Minute extracted from timestamp'
            factors_info['second'] = 'Second extracted from timestamp'

        df = df.set_index('timestamp')

        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    def _cal_label(self, df):
        """
        Categorical target for classification tasks.
        """
        df = deepcopy(df)
        labels_info = {}

        end = self.future_timestamps + 1 if self.if_use_future else 2

        for i in range(1, end):
            df[f"ret{i:04d}"] = df['close'].pct_change(i).shift(-i)
            df[f"ret{i:04d}"] = df[f"ret{i:04d}"].fillna(0.0)
            df[f"mov{i:04d}"] = (df[f"ret{i:04d}"] > 0)
            df[f"mov{i:04d}"] = df[f"mov{i:04d}"].fillna(0).astype(int)
            labels_info[f"ret{i:04d}"] = {
                "description": f"Return {i:04d} days later",
            }
            labels_info[f"mov{i:04d}"] = {
                "description": f"Movement {i:04d} days later, 1 for up, 0 for down",
            }
        df = df[[col for col in labels_info.keys()]].copy()
        return df, labels_info

    def _load_asset_dataframe(self):
        """
        Load asset dataframes from the splits.
        :return: Dict[str, pd.DataFrame]
        """
        # Load price
        enabled_price_info = self.enabled_data_info["prices_data_info"][0]
        price_df = self._load_dataframe(enabled_price_info, self.symbol)
        price_columns = list(sorted(price_df.columns.tolist()))
        price_df = price_df[price_columns]

        # Calculate labels
        label_df, _ = self._cal_label(price_df)
        label_columns = list(sorted(label_df.columns.tolist()))
        label_df = label_df[label_columns]

        # Calculate times
        time_df = None
        time_columns = None
        if self.if_use_temporal:
            time_df, _ = self._cal_time(price_df, level=TimeLevel.from_string(enabled_price_info["level"]))
            time_columns = list(sorted(time_df.columns.tolist()))
            time_df = time_df[time_columns]

        # Calculate features
        feature_df = None
        feature_columns = None
        if self.if_use_features:
            enabled_feature_info = self.enabled_data_info["features_data_info"][0]
            feature_df = self._load_dataframe(enabled_feature_info, symbol=self.symbol)
            feature_df[price_columns] = price_df
            feature_columns = list(sorted(feature_df.columns.tolist()))
            feature_df = feature_df[feature_columns]

        # Load news
        news_df = None
        news_columns = None
        if self.if_use_news:
            enabled_news_info = self.enabled_data_info["news_data_info"]
            news_df = self._load_dataframe(enabled_news_info, symbol=self.symbol)
            news_columns = list(sorted(news_df.columns.tolist()))
            news_df = news_df[news_columns]

        dfs = dict(
            price_df=price_df,
            price_columns=price_columns,
            time_df=time_df,
            time_columns=time_columns,
            feature_df=feature_df,
            feature_columns=feature_columns,
            label_df=label_df,
            label_columns=label_columns,
            news_df=news_df,
            news_columns=news_columns,
        )

        return dfs

    def _init_asset_data(self):

        dfs = self._load_asset_dataframe()

        price_df = dfs["price_df"]
        price_columns = dfs["price_columns"]
        time_df = dfs["time_df"]
        time_columns = dfs["time_columns"]
        feature_df = dfs["feature_df"]
        feature_columns = dfs["feature_columns"]
        label_df = dfs["label_df"]
        label_columns = dfs["label_columns"]
        news_df = dfs["news_df"]
        news_columns = dfs["news_columns"]

        original_prices = deepcopy(price_df) # Original prices before scaling

        price_scaler = None
        feature_scaler = None
        time_scaler = None
        if self.if_norm:
            # Scale prices
            price_scaler = SCALER.build(self.scaler_cfg)
            price_df = price_scaler.fit_transform(price_df)

            if len(price_scaler.mean.shape) == 1 and len(price_scaler.std.shape) == 1:
                prices_mean = price_scaler.mean[None, 1].repeat(price_df.shape[0], axis=0)
                prices_std = price_scaler.std[None, 1].repeat(price_df.shape[0], axis=0)
            else:
                prices_mean = price_scaler.mean
                prices_std = price_scaler.std

            if self.if_use_temporal:
                if self.if_norm_temporal:
                    time_scaler = SCALER.build(self.scaler_cfg)
                    time_df = time_scaler.fit_transform(time_df)

            if self.if_use_features:
                feature_scaler = SCALER.build(self.scaler_cfg)
                feature_df = feature_scaler.fit_transform(feature_df)

        else:
            prices_mean = np.zeros((price_df.shape[0], len(price_df.columns)))
            prices_std = np.ones((price_df.shape[0], len(price_df.columns)))

        prices_mean_df = pd.DataFrame(prices_mean, index=price_df.index, columns=price_columns)
        prices_std_df = pd.DataFrame(prices_std, index=price_df.index, columns=price_columns)

        prices = price_df # Scaled prices
        times = time_df if self.if_use_temporal else None
        features = feature_df if self.if_use_features else None
        labels = label_df
        news = news_df if self.if_use_news else None
        prices_mean = prices_mean_df
        prices_std = prices_std_df

        data = dict(
            symbol=self.symbol,
            prices=prices,
            price_columns=price_columns,
            times=times,
            time_columns=time_columns,
            features=features,
            feature_columns=feature_columns,
            labels=labels,
            label_columns=label_columns,
            news=news,
            news_columns=news_columns,
            original_prices=original_prices,
            prices_mean=prices_mean,
            prices_std=prices_std,
            price_scaler=price_scaler,
            feature_scaler=feature_scaler,
            time_scaler=time_scaler,
        )

        return data

    def _init_asset_meta_info(self):
        future_timestamps = self.future_timestamps

        count = 0
        prices = self.asset_data["prices"]
        asset_info = self.asset_info_from_file

        items: Dict[int, Any] = {}
        for i in range(self.history_timestamps, len(prices) - future_timestamps):

            id = count
            items[id]: Dict[str, Any] = {}

            history_df = prices.iloc[i - self.history_timestamps: i]

            start_timestamp = history_df.index[0]
            end_timestamp = history_df.index[-1]
            start_timestamp, end_timestamp = get_start_end_timestamp(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                level=self.level
            )

            history_info = {
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "start_index": i - self.history_timestamps,
                "end_index": i - 1,
            }

            items[id].update(
                {"history_info": history_info}
            )

            if self.if_use_future:

                future_df = prices.iloc[i: i + self.future_timestamps]

                start_timestamp = future_df.index[0]
                end_timestamp = future_df.index[-1]

                start_timestamp, end_timestamp = get_start_end_timestamp(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    level=self.level
                )

                future_info = {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "start_index": i,
                    "end_index": i + self.future_timestamps - 1,
                }

                items[id].update(
                    {"future_info": future_info}
                )

            count += 1

        shape = dict(
            prices=prices.shape,
            labels=self.asset_data["labels"].shape,
            times=self.asset_data["times"].shape if self.if_use_temporal else None,
            features=self.asset_data["features"].shape if self.if_use_features else None,
            news=self.asset_data["news"].shape if self.if_use_news else None,
            original_prices=self.asset_data["original_prices"].shape,
            prices_mean=self.asset_data["prices_mean"].shape,
            prices_std=self.asset_data["prices_std"].shape,
        )

        columns = dict(
            prices=self.asset_data["price_columns"],
            labels=self.asset_data["label_columns"],
            times=self.asset_data["time_columns"] if self.if_use_temporal else None,
            features=self.asset_data["feature_columns"] if self.if_use_features else None,
            news=self.asset_data["news_columns"] if self.if_use_news else None,
        )

        meta_info = dict(
            symbol=self.symbol,
            asset_info=asset_info,
            shape=shape,
            columns=columns,
            items=items,
            length=count
        )

        return meta_info

    def crop(self,
            start_timestamp: str = None,
            end_timestamp: str = None,
            ):

        start_timestamp = pd.to_datetime(start_timestamp)
        end_timestamp = pd.to_datetime(end_timestamp)

        timestamp_info = {}
        assets_meta_info = self.asset_meta_info['items']
        for key, value in assets_meta_info.items():
            start_timestamp_ = value["history_info"]["start_timestamp"]
            end_timestamp_ = value["history_info"]["end_timestamp"]

            if (end_timestamp_ >= start_timestamp
                    and end_timestamp_ <= end_timestamp):
                timestamp_info[key] = {
                    "start_timestamp": start_timestamp_,
                    "end_timestamp": end_timestamp_,
                }

        timestamp_min_index = min(timestamp_info.keys())
        start_timestamp = timestamp_info[timestamp_min_index]["start_timestamp"]

        asset_data = self.asset_data
        sub_data = dict(
            prices=self._get_dataitem(asset_data["prices"], start_timestamp, end_timestamp),
            times=self._get_dataitem(asset_data["times"], start_timestamp, end_timestamp) if self.if_use_temporal else None,
            features=self._get_dataitem(asset_data["features"], start_timestamp, end_timestamp) if self.if_use_features else None,
            labels=self._get_dataitem(asset_data["labels"], start_timestamp, end_timestamp),
            news=self._get_dataitem(asset_data["news"], start_timestamp, end_timestamp) if self.if_use_news else None,
            original_prices=self._get_dataitem(asset_data["original_prices"], start_timestamp, end_timestamp),
            prices_mean=self._get_dataitem(asset_data["prices_mean"], start_timestamp, end_timestamp),
            prices_std=self._get_dataitem(asset_data["prices_std"], start_timestamp, end_timestamp),
        )
        asset_data.update(sub_data)
        self.asset_data = asset_data
        self.asset_info = self.asset_info_from_file
        self.asset_meta_info = self._init_asset_meta_info()

    def __str__(self):
        str = f"{'-' * 50} SingleAssetDataset {'-' * 50}\n"
        str += f"- Asset: {self.symbol}\n"
        str += f"- Length: {self.asset_meta_info['length']}\n"

        shape_str = "\n".join([f"\t{key}: {value}" for key, value in self.asset_meta_info["shape"].items()])
        str += f"- Shape: \n {shape_str}\n"

        columns_str = "\n".join([f"\t{key}: {value}" for key, value in self.asset_meta_info["columns"].items()])
        str += f"- Columns: \n{columns_str}\n"

        str += f"{'-' * 50} SingleAssetDataset {'-' * 50}\n"
        return str

    def __len__(self):
        return self.asset_meta_info["length"]

    def _get_dataitem(self,
                      df: DataFrame,
                      start_timestamp: Any,
                      end_timestamp: Any):
        df = deepcopy(df)
        df = df[(start_timestamp <= df.index) & (df.index <= end_timestamp)]
        return df

    def __getitem__(self, idx):

        id = idx
        item = self.asset_meta_info['items'][id]
        asset = self.symbol
        res = dict(
            asset=asset,
        )

        history_info = item["history_info"]

        start_timestamp = history_info["start_timestamp"]
        end_timestamp = history_info["end_timestamp"]
        start_index = history_info["start_index"]
        end_index = history_info["end_index"]
        history_data = {
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "start_index": start_index,
            "end_index": end_index,
            "prices": self._get_dataitem(self.asset_data["prices"], start_timestamp, end_timestamp).values.astype("float32"),
            "labels": self._get_dataitem(self.asset_data["labels"], start_timestamp, end_timestamp).values.astype("float32"),
            "original_prices": self._get_dataitem(self.asset_data["original_prices"], start_timestamp, end_timestamp).values.astype("float32"),
            "timestamps": self._get_dataitem(self.asset_data["prices"], start_timestamp, end_timestamp).reset_index(drop=False)["timestamp"].tolist(),
            "prices_mean": self._get_dataitem(self.asset_data["prices_mean"], start_timestamp, end_timestamp).values.astype("float32"),
            "prices_std": self._get_dataitem(self.asset_data["prices_std"], start_timestamp, end_timestamp).values.astype("float32"),
        }
        if self.if_use_features:
            history_data["features"] = self._get_dataitem(self.asset_data["features"], start_timestamp, end_timestamp).values.astype("float32")
        if self.if_use_temporal:
            history_data["times"] = self._get_dataitem(self.asset_data["times"], start_timestamp, end_timestamp).values.astype("float32")
        if self.if_use_news:
            # Filter news data based on the timestamps
            history_data["news"] = self._get_dataitem(self.asset_data["news"], start_timestamp, end_timestamp)

        res["history"] = history_data

        if self.if_use_future:

            future_info = item["future_info"]

            start_timestamp = future_info["start_timestamp"]
            end_timestamp = future_info["end_timestamp"]
            start_index = future_info["start_index"]
            end_index = future_info["end_index"]
            future_data = {
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "start_index": start_index,
                "end_index": end_index,
                "prices": self._get_dataitem(self.asset_data["prices"], start_timestamp, end_timestamp).values.astype("float32"),
                "labels": self._get_dataitem(self.asset_data["labels"], start_timestamp, end_timestamp).values.astype("float32"),
                "original_prices": self._get_dataitem(self.asset_data["original_prices"], start_timestamp, end_timestamp).values.astype("float32"),
                "timestamps": self._get_dataitem(self.asset_data["prices"], start_timestamp, end_timestamp).reset_index(drop=False)["timestamp"].tolist(),
                "prices_mean": self._get_dataitem(self.asset_data["prices_mean"], start_timestamp, end_timestamp).values.astype("float32"),
                "prices_std": self._get_dataitem(self.asset_data["prices_std"], start_timestamp, end_timestamp).values.astype("float32"),
            }

            if self.if_use_features:
                future_data["features"] = self._get_dataitem(self.asset_data["features"], start_timestamp, end_timestamp).values.astype("float32")
            if self.if_use_temporal:
                future_data["times"] = self._get_dataitem(self.asset_data["times"], start_timestamp, end_timestamp).values.astype("float32")
            if self.if_use_news:
                # Filter news data based on the timestamps
                future_data["news"] = self._get_dataitem(self.asset_data["news"], start_timestamp, end_timestamp)

            res["future"] = future_data

        return res

__all__ = [
    "SingleAssetDataset"
]

if __name__ == '__main__':

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
    dataset = DATASET.build(dataset)
    print(dataset)

    subdataset = deepcopy(dataset)
    subdataset.crop(
        start_timestamp="2015-05-01",
        end_timestamp="2023-05-01"
    )
    print(subdataset)

    collate_fn = SingleAssetPriceTextCollateFn()
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn)
    item = next(iter(dataloader))
    history = item["history"]
    print(history["prices"].shape)
    print(history["features"].shape)
    print("times", history['times'])
    print(history["start_timestamp"])
    print(history["end_timestamp"])
    print(history["timestamps"])