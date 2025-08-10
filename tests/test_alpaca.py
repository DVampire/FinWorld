import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)
import pandas as pd

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.log import logger
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime

async def price():
    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )

    request_params = StockBarsRequest(
        symbol_or_symbols="AAPL",
        timeframe=TimeFrame.Minute,
        start=datetime.strptime("2025-05-01", '%Y-%m-%d'),
        end=datetime.strptime("2025-05-07", '%Y-%m-%d')
    )

    bars = client.get_stock_bars(request_params)

    df = bars.df
    df = df.reset_index()

    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    return df

def get_news():
    client = NewsClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY")
    )

    request_params = NewsRequest(
        symbols="AAPL",
        start=datetime.strptime("2025-05-01", '%Y-%m-%d'),
        end=datetime.strptime("2025-05-07", '%Y-%m-%d'),
        include_content=True,
    )
    news = client.get_news(request_params)
    df = news.df
    df['timestamp'] = pd.to_datetime(df['created_at'])

    print(df.columns)

    df = df.reset_index()
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    return df

if __name__ == "__main__":

    # Initialize the logger
    log_path = os.path.join(root, "workdir", "tests")
    os.makedirs(log_path, exist_ok=True)

    log_path = os.path.join(log_path, "logger.log")
    logger.init_logger(log_path=log_path, accelerator=None)

    # price = asyncio.run(price())
    # logger.info(price)

    news = get_news()
    for i in range(10):
        logger.info(news.iloc[i].to_dict())