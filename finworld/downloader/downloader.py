import asyncio
import os
import json
from typing import Optional, Any
from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.downloader.custom import AbstractDownloader
from finworld.utils import assemble_project_path
from finworld.downloader.fmp import FMPPriceDownloader, FMPNewsDownloader, FMPSymbolInfoDownloader
from finworld.downloader.alpaca import AlpacaPriceDownloader, AlpacaNewsDownloader
from finworld.downloader.akshare import AkSharePriceDownloader, AkShareNewsDownloader
from finworld.downloader.tushare import TuSharePriceDownloader, TuShareNewsDownloader
from finworld.registry import DOWNLOADER
from finworld.config import config
from finworld.log import logger

@DOWNLOADER.register_module(force=True)
class PriceDownloader(AbstractDownloader):
    def __init__(self,
                 source: str = "fmp",
                 assets_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__()

        self.source = source
        self.assets_path = assemble_project_path(assets_path)
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format

        self.assets_info, self.symbols = self._load_assets()

        assert len(self.symbols) > 0, "No symbols to download"
        self.max_concurrent = max_concurrent

        self.exp_path = config.exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _load_assets(self):
        """
        Load assets from the assets file.
        :return:
        """
        with open(self.assets_path) as f:
            assets_info = json.load(f)
        symbols = [asset for asset in assets_info]
        logger.info(f"| Loaded {len(symbols)} assets from {self.assets_path}")
        return assets_info, symbols

    def _get_sybmol_info(self, symbol: str) -> Any:
        """
        Get symbol info from the assets file.
        :param symbol:
        :return:
        """
        if symbol in self.assets_info:
            return self.assets_info[symbol]
        else:
            raise ValueError(f"Symbol {symbol} not found in assets file")

    async def run_task(self, task: Any):
        await task.run()

    async def _download_fmp_price(self, save_dir: str):
        """
        Download price data from FMP API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = FMPPriceDownloader(
                api_key=os.getenv("FMP_API_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_alpaca_price(self, save_dir: str):
        """
        Download price data from Alpaca API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = AlpacaPriceDownloader(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_akshare_price(self, save_dir: str):
        """
        Download price data from AkShare API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = AkSharePriceDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_tushare_price(self, save_dir: str):
        """
        Download price data from TuShare API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = TuSharePriceDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def run(self):
        if self.source == "fmp":
            logger.info(f"| Downloading price data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "price")
            await self._download_fmp_price(save_dir = save_dir)
        elif self.source == "alpaca":
            logger.info(f"| Downloading price data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "price")
            await self._download_alpaca_price(save_dir = save_dir)
        elif self.source == "akshare":
            logger.info(f"| Downloading price data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "price")
            await self._download_akshare_price(save_dir = save_dir)
        elif self.source == "tushare":
            logger.info(f"| Downloading price data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "price")
            await self._download_tushare_price(save_dir = save_dir)

@DOWNLOADER.register_module(force=True)
class NewsDownloader(AbstractDownloader):
    def __init__(self,
                 source: str = "fmp",
                 assets_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__()

        self.source = source
        self.assets_path = assemble_project_path(assets_path)
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format

        self.assets_info, self.symbols = self._load_assets()

        assert len(self.symbols) > 0, "No symbols to download"
        self.max_concurrent = max_concurrent

        self.exp_path = config.exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _load_assets(self):
        """
        Load assets from the assets file.
        :return:
        """
        with open(self.assets_path) as f:
            assets_info = json.load(f)
        symbols = [asset for asset in assets_info]
        logger.info(f"| Loaded {len(symbols)} assets from {self.assets_path}")
        return assets_info, symbols

    def _get_sybmol_info(self, symbol: str) -> Any:
        """
        Get symbol info from the assets file.
        :param symbol:
        :return:
        """
        if symbol in self.assets_info:
            return self.assets_info[symbol]
        else:
            raise ValueError(f"Symbol {symbol} not found in assets file")

    async def run_task(self, task: Any):
        await task.run()

    async def _download_fmp_news(self, save_dir: str):
        """
        Download news data from FMP API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = FMPNewsDownloader(
                api_key=os.getenv("FMP_API_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_alpaca_news(self, save_dir: str):
        """
        Download news data from Alpaca API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = AlpacaNewsDownloader(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_akshare_news(self, save_dir: str):
        """
        Download news data from AkShare API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = AkShareNewsDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def _download_tushare_news(self, save_dir: str):
        """
        Download news data from TuShare API.
        :return:
        """

        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            downloader = TuShareNewsDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir
            )
            tasks.append(downloader)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

    async def run(self):
        if self.source == "fmp":
            logger.info(f"| Downloading news data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "news")
            await self._download_fmp_news(save_dir = save_dir)

        elif self.source == "alpaca":
            logger.info(f"| Downloading news data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "news")
            await self._download_alpaca_news(save_dir = save_dir)

        elif self.source == "akshare":
            logger.info(f"| Downloading news data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "news")
            await self._download_akshare_news(save_dir = save_dir)
        elif self.source == "tushare":
            logger.info(f"| Downloading news data from {self.source}...")
            save_dir = os.path.join(self.exp_path, "news")
            await self._download_tushare_news(save_dir = save_dir)

@DOWNLOADER.register_module(force=True)
class SymbolInfoDownloader(AbstractDownloader):
    """Downloader for fetching symbol information from various sources."""
    def __init__(self,
                 source: str = "fmp",
                 save_name: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.source = source
        self.exp_path = config.exp_path
        self.save_name = save_name if save_name else "full"
        self.max_concurrent = max_concurrent if max_concurrent else 10
        os.makedirs(self.exp_path, exist_ok=True)

    async def run(self):
        if self.source == "fmp":
            logger.info(f"| Downloading symbol information from {self.source}...")
            save_dir = os.path.join(self.exp_path, "symbol_info")
            downloader = FMPSymbolInfoDownloader(
                api_key=os.getenv("FMP_API_KEY"),
                save_name=self.save_name,
                exp_path=save_dir
            )
            await downloader.run()
        else:
            raise ValueError(f"Source {self.source} is not supported for symbol information download.")