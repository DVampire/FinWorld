import os

from typing import Optional, Any
from datetime import datetime
import pandas as pd
from markdownify import markdownify
import akshare as ak

from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.downloader.custom import AbstractDownloader
from finworld.log import logger
from finworld.calendar import calendar_manager

class AkSharePriceDownloader(AbstractDownloader):
    def __init__(self,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the DataFrame to ensure consistent column order and types.
        :param df: DataFrame to format.
        :return: Formatted DataFrame.
        """
        if len(df)> 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp", ascending=True)
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]
        return df

    async def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            symbol_info: Optional[Any] = None,
            ):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        symbol_info = symbol_info if symbol_info else self.symbol_info
        symbol = symbol_info["symbol"]
        exchange = symbol_info.get("exchange", 'Shanghai Stock Exchange')

        start_date = start_date.strftime("%Y%m%d")
        end_date = end_date.strftime("%Y%m%d")

        if self.level == "1day":
            level = "daily"
            df = ak.stock_zh_a_hist(symbol=symbol.split('.')[0],
                                    period=level,
                                    start_date=start_date,
                                    end_date=end_date)

            map_columns = {
                '日期': 'timestamp',
                '股票代码': 'symbol',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover_rate'
            }
            df = df.rename(columns=map_columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d")
            df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Shanghai')

        elif self.level == "1hour":
            raise NotImplementedError("Hourly data is not supported by Akshare for A-shares.")
        elif self.level == "1min":
            self.level = "minute"
            raise NotImplementedError("Minute data is not supported by Akshare for A-shares.")

        df = calendar_manager.convert_timezone(symbol_info=symbol_info, dataframe=df)

        df = df.drop(columns=['symbol'])
        df = self._format_dataframe(df)

        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)

        logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")

class AkShareNewsDownloader(AbstractDownloader):
    def __init__(self,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)
    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the DataFrame to ensure consistent column order and types.
        :param df: DataFrame to format.
        :return: Formatted DataFrame.
        """
        if len(df)> 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by="timestamp", ascending=True)
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]
        return df

    async def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            symbol_info: Optional[Any] = None,
            ):
        raise NotImplementedError("AkShare News Downloader is not implemented yet.")