import os
import pandas as pd
from typing import Optional, Any
from datetime import datetime

from finworld.processor.custom import AbstractProcessor
from finworld.registry import FACTOR

class AkSharePriceProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 feature_type: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.feature_type = feature_type

        self.exp_path = exp_path

    async def run(self):
        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        # update df to info
        info["names"] = list(sorted(df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info

class AkShareFeatureProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 feature_type: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None

        self.exp_path = exp_path

        self.factor_method = FACTOR.build(
            dict(
                type=feature_type,
                windows=[5, 10, 20, 30, 60],
                level=self.level,
            ))

    async def run(self):

        info = {
            "symbol": self.symbol,
        }

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        price_column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        data_path = os.path.join(self.data_path, "{}.jsonl".format(self.symbol))

        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        assert os.path.exists(data_path), "Price path {} does not exist".format(data_path)

        df = pd.read_json(data_path, lines=True)

        df = df.rename(columns=price_column_map)[["timestamp"] + price_columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp")
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
        df = df.reset_index(drop=True)
        df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        res = await self.factor_method.run(df)

        factors_df = res["factors_df"]
        factors_info = res["factors_info"]

        info["names"] = list(sorted(factors_df.columns))
        info["start_date"] = df["timestamp"].min()
        info["end_date"] = df["timestamp"].max()

        factors_df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(self.symbol)), orient="records", lines=True)

        return info

class AkShareNewsProcessor(AbstractProcessor):
    def __init__(self,
                 data_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 feature_type: Optional[str] = None,
                 ):
        super().__init__()

        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent
        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.exp_path = exp_path
        self.feature_type = feature_type

    async def run(self):
        raise NotImplementedError("AkShareNewsProcessor is not implemented yet.")
