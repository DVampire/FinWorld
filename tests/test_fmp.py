import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.log import logger
from finworld.utils import get_jsonparsed_data

async def test_fmp_day():
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&from=2025-05-01&to=2025-05-01&apikey={os.getenv('FMP_API_KEY')}"
    data = await get_jsonparsed_data(url)
    logger.info(f"Data for AAPL: {data}")

async def test_fmp_minute():
    url = f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=AAPL&from=2025-04-30&to=2025-04-30&apikey={os.getenv('FMP_API_KEY')}"
    data = await get_jsonparsed_data(url)
    logger.info(f"Data for AAPL: {data}")

async def test_fmp_news():
    url = f"https://financialmodelingprep.com/stable/news/stock?symbols=AAPL&from=2010-05-01&to=2011-05-01&limit=250&page=0&apikey={os.getenv('FMP_API_KEY')}"
    data = await get_jsonparsed_data(url)
    logger.info(f"News for AAPL: {data}")

if __name__ == "__main__":

    # Initialize the logger
    log_path = os.path.join(root, "workdir", "tests")
    os.makedirs(log_path, exist_ok=True)

    log_path = os.path.join(log_path, "logger.log")
    logger.init_logger(log_path=log_path, accelerator=None)

    # asyncio.run(test_fmp_day())
    # asyncio.run(test_fmp_minute())
    asyncio.run(test_fmp_news())