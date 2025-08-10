import os

from typing import Optional, Any
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pandas as pd
from markdownify import markdownify

from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.downloader.custom import AbstractDownloader
from finworld.log import logger
from finworld.calendar import calendar_manager

class AlpacaPriceDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.api_key = api_key
        self.secret_key = secret_key

        if self.api_key is None:
            self.api_key = os.getenv("ALPACA_API_KEY")
        if self.secret_key is None:
            self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

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

        start_date = datetime.strptime(start_date if start_date
                                       else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date
                                     else self.end_date, "%Y-%m-%d")

        symbol_info = symbol_info if symbol_info else self.symbol_info
        symbol = symbol_info["symbol"]
        exchange = symbol_info.get("exchange", 'New York Stock Exchange')

        timeframe = None
        if "day" in self.level:
            amount = int(self.level.replace("day", ""))
            timeframe = TimeFrame(amount, TimeFrameUnit.Day)
        elif "min" in self.level:
            amount = int(self.level.replace("min", ""))
            timeframe = TimeFrame(amount, TimeFrameUnit.Minute)
        elif "hour" in self.level:
            amount = int(self.level.replace("hour", ""))
            timeframe = TimeFrame(amount, TimeFrameUnit.Hour)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request_params)

        df = bars.df
        df = df.reset_index()

        df = calendar_manager.convert_timezone(symbol_info=symbol_info, dataframe=df)

        df = df.drop(columns=['symbol'])
        df = self._format_dataframe(df)

        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)

        logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")


class AlpacaNewsDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        self.api_key = api_key
        self.secret_key = secret_key

        if self.api_key is None:
            self.api_key = os.getenv("ALPACA_API_KEY")
        if self.secret_key is None:
            self.secret_key = os.getenv("ALPACA_SECRET_KEY")

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

        self.client = NewsClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

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

        start_date = datetime.strptime(start_date if start_date
                                       else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date
                                     else self.end_date, "%Y-%m-%d")

        symbol_info = symbol_info if symbol_info else self.symbol_info
        symbol = symbol_info["symbol"]
        exchange = symbol_info.get("exchange", 'New York Stock Exchange')

        request_params = NewsRequest(
            symbols=symbol,
            start=start_date,
            end=end_date,
            include_content=True,
        )

        news = self.client.get_news(request_params)

        df = news.df
        df['timestamp'] = pd.to_datetime(df['created_at'])
        df = df.reset_index()

        df = calendar_manager.convert_timezone(symbol_info=symbol_info, dataframe = df)

        # process the content
        df['raw_content'] = df['content']
        df['content'] = df['raw_content'].apply(lambda x: markdownify(x) if isinstance(x, str) else x)
        df['title'] = df['headline']

        df = df.drop(columns=['symbols', 'headline'])
        df = self._format_dataframe(df)

        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)

        logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")