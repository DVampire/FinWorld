import asyncio
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple, Any
import json

from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.downloader.custom import AbstractDownloader
from finworld.utils import (get_jsonparsed_data,
                            generate_intervals,
                            fetch_url,
                            get_newspage_name)
from finworld.log import logger
from finworld.calendar import calendar_manager

class FMPPriceDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
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

        if self.api_key is None:
            self.api_key = os.getenv("FMP_API_KEY")

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent

        if "day" in level:
            self.request_url = "https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={}&from={}&to={}&apikey={}"
            self.interval_level = "year"
        elif "min" in level or "hour" in level:
            self.request_url = f"https://financialmodelingprep.com/stable/historical-chart/{self.level}?" + "symbol={}&from={}&to={}&apikey={}"
            self.interval_level = "day"

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _check_download(self,
                        symbol: Optional[str] = None,
                        intervals: Optional[List[Tuple[datetime, datetime]]] = None
                        ):

        download_infos = []

        for (start, end) in intervals:
            name = f"{start.strftime('%Y-%m-%d')}.jsonl"
            if os.path.exists(os.path.join(self.exp_path, symbol, name)):
                item = {
                    "name": name,
                    "downloaded": True,
                    "start": start,
                    "end": end
                }
            else:
                item = {
                    "name": name,
                    "downloaded": False,
                    "start": start,
                    "end": end
                }
            download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)

        logger.info(f"| {self.symbol} Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")

        return download_infos

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

    async def run_task(self, task: Any):
        """
        Run a single download task.
        :param task: Dictionary containing the task details.
        :return: None
        """
        url = task["url"]
        save_path = task["save_path"]
        columns = task["columns"]

        df = {column:[] for column in columns}

        try:
            await asyncio.sleep(1)  # To avoid hitting the API too fast
            aggs = await get_jsonparsed_data(url)
        except Exception as e:
            logger.error("| Download Failed: {}, Error: {}".format(save_path, e))
            aggs = []

        if len(aggs) == 0:
            logger.error(f"| Download Failed: {save_path}. No data found for {url}")
            return

        for a in aggs:
            df["timestamp"].append(a["date"])
            df["open"].append(a["open"])
            df["high"].append(a["high"])
            df["low"].append(a["low"])
            df["close"].append(a["close"])
            df["volume"].append(a["volume"])

            if "day" in self.level:
                df["change"].append(a["change"])
                df["changePercent"].append(a["changePercent"])
                df["vwap"].append(a["vwap"])

        df = pd.DataFrame(df, index=range(len(df["timestamp"])))
        df = self._format_dataframe(df)
        df.to_json(save_path, orient="records", lines=True)
        logger.info(f"| Downloaded Success: {save_path}.")

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

        intervals = generate_intervals(start_date, end_date, self.interval_level)
        valid_days = calendar_manager.get_valid_days(symbol_info, start_date=start_date, end_date=end_date)

        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )

        save_dir = os.path.join(self.exp_path, symbol)
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for info in download_infos:

            name = info["name"]
            downloaded = info["downloaded"]
            start = info["start"]
            end = info["end"]

            if self.interval_level == "year":
                is_trading_day = True
            elif self.interval_level == "day":
                is_trading_day = pd.DatetimeIndex([start], dtype="datetime64[ns, UTC]")[0] in valid_days
            else:
                is_trading_day = True
            if is_trading_day and not downloaded:
                if "day" in self.level:
                    columns = ["timestamp", "open", "high", "low", "close", "volume", "change", "changePercent", "vwap"]
                else:
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]

                request_url = self.request_url.format(
                    symbol,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                    self.api_key
                )
                save_path = os.path.join(save_dir, name)

                task = {
                    "url": request_url,
                    "save_path": save_path,
                    "columns": columns,
                }
                tasks.append(task)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

        # After all tasks are done, we can read the downloaded files and concatenate them
        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )
        df = pd.DataFrame()
        for info in download_infos:
            name = info["name"]
            downloaded = info["downloaded"]

            if downloaded:
                chunk_df = pd.read_json(os.path.join(save_dir, name), lines=True)
                df = pd.concat([df, chunk_df], axis=0)
        df = self._format_dataframe(df)
        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)

        logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")

class FMPNewsDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 symbol_info: Optional[Any] = None,
                 exp_path: Optional[str] = None,
                 max_pages: int = 100,
                 fetch_url: Optional[bool] = False,
                 **kwargs):
        super().__init__()
        self.api_key = api_key

        if self.api_key is None:
            self.api_key = os.getenv("FMP_API_KEY")

        self.symbol_info = symbol_info
        self.symbol = symbol_info["symbol"] if symbol_info else None
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format
        self.max_concurrent = max_concurrent
        self.max_pages = max_pages
        self.fetch_url = fetch_url

        self.old_request_url = "https://financialmodelingprep.com/api/v3/stock_news?tickers={}&page={}&limit=100&from={}&to={}&apikey={}"
        self.request_url = "https://financialmodelingprep.com/stable/news/stock?symbols={}&page={}&limit=250&from={}&to={}&apikey={}"

        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    def _check_download(self,
                        symbol: Optional[str] = None,
                        intervals: Optional[List[Tuple[datetime, datetime]]] = None):

        download_infos = []

        for (start, end) in intervals:
            for page in range(0, self.max_pages):
                name = f"{start.strftime('%Y-%m-%d')}_page_{page:04d}.jsonl"
                if os.path.exists(os.path.join(self.exp_path, symbol, name)):
                    item = {
                        "name": name,
                        "downloaded": True,
                        "start": start,
                        "end": end,
                        "page": page
                    }
                else:
                    item = {
                        "name": name,
                        "downloaded": False,
                        "start": start,
                        "end": end,
                        "page": page
                    }
                download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)

        logger.info(f"| {self.symbol} News Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")

        return download_infos

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

    async def run_task(self, task: Any):
        """
        Run a single download task.
        :param task: Dictionary containing the task details.
        :return: None
        """
        old_url = task["old_url"]
        url = task["url"]
        save_path = task["save_path"]
        columns = task["columns"]

        df = {column:[] for column in columns}

        aggs = []
        try:
            await asyncio.sleep(1)  # To avoid hitting the API too fast
            aggs1 = await get_jsonparsed_data(url)
            aggs2 = await get_jsonparsed_data(old_url)
            if len(aggs1) > 0:
                aggs.extend(aggs1)
            if len(aggs2) > 0:
                aggs.extend(aggs2)
        except Exception as e:
            logger.error("| Download Failed: {}, Error: {}".format(save_path, e))

        if len(aggs) == 0:
            logger.error(f"| Download Failed: {save_path}. No data found for {url}")
            df = pd.DataFrame(df, index=range(len(df["timestamp"])))
            df.to_json(save_path, orient="records", lines=True)
            return

        for a in aggs:
            df["timestamp"].append(a["publishedDate"])
            df["title"].append(a["title"])
            df["image"].append(a["image"])
            df["site"].append(a["site"])
            df["raw_content"].append(a["text"])
            df["url"].append(a["url"])

        df = pd.DataFrame(df, index=range(len(df["timestamp"])))
        df = self._format_dataframe(df)
        df.to_json(save_path, orient="records", lines=True)
        logger.info(f"| Downloaded Success: {save_path}.")


    async def fetch_task(self, task: dict):
        """
        Fetch the markdown content for a news item.
        :param task: Dictionary containing the task details.
        :return: None
        """
        name = task["name"]
        downloaded = task["downloaded"]
        url = task["url"]
        save_path = task["save_path"]

        if downloaded:
            logger.info(f"| Markdown already downloaded: {name}")
            return

        content = await fetch_url(url)
        if content:
            with open(save_path, "w") as f:
                f.write(content.markdown)
            logger.info(f"| Markdown downloaded and saved: {save_path}")
        else:
            logger.error(f"| Failed to fetch markdown content for {name}")

    async def download_markdown(self, df: pd.DataFrame)-> pd.DataFrame:
        save_path = os.path.join(os.path.dirname(self.exp_path), "markdown", self.symbol)
        os.makedirs(save_path, exist_ok=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

        tasks = []
        for row in df.iterrows():
            row = row[1]
            timestamp = row["timestamp"]
            title = row["title"]
            symbol = self.symbol

            name = get_newspage_name(
                symbol=symbol,
                timestamp=timestamp,
                title=title
            )
            name = f"{name}.md"

            markdown_path = os.path.join(save_path, name)
            if os.path.exists(markdown_path):
                task = {
                    "name": name,
                    "downloaded": True,
                    "url": row["url"],
                    "save_path": markdown_path,
                }
            else:
                task = {
                    "name": name,
                    "downloaded": False,
                    "url": row["url"],
                    "save_path": markdown_path,
                }
            tasks.append(task)

        logger.info(f"| Downloading markdown content for {len(tasks)} news items...")
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.fetch_task(task) for task in batch])

        # check if all markdown files are downloaded
        markdown_paths = []
        contents = []
        for task in tasks:
            name = task["name"]
            save_path = task["save_path"]
            if os.path.exists(save_path):
                markdown_paths.append(name)
                with open(save_path, "r") as f:
                    content = f.read()
                contents.append(content)
            else:
                markdown_paths.append("")
                contents.append("")

        df['content'] = contents
        df['markdown_path'] = markdown_paths

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

        intervals = generate_intervals(start_date, end_date, "year")
        valid_days = calendar_manager.get_valid_days(symbol_info, start_date=start_date, end_date=end_date)

        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )

        save_dir = os.path.join(self.exp_path, symbol)
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for info in download_infos:

            name = info["name"]
            downloaded = info["downloaded"]
            start = info["start"]
            end = info["end"]
            page = info["page"]

            is_trading_day = pd.DatetimeIndex([start], dtype="datetime64[ns, UTC]")[0] in valid_days
            if is_trading_day and not downloaded:
                columns = ["timestamp", "title", "image", "site", "raw_content", "url"]

                old_request_url = self.old_request_url.format(
                    symbol,
                    page,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                    self.api_key
                )

                request_url = self.request_url.format(
                    symbol,
                    page,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                    self.api_key
                )
                save_path = os.path.join(save_dir, name)

                task = {
                    "old_url": old_request_url,
                    "url": request_url,
                    "save_path": save_path,
                    "columns": columns,
                }
                tasks.append(task)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            await asyncio.gather(*[self.run_task(task) for task in batch])

        # After all tasks are done, we can read the downloaded files and concatenate them
        download_infos = self._check_download(
            symbol=symbol,
            intervals=intervals,
        )
        df = pd.DataFrame()
        for info in download_infos:
            name = info["name"]
            downloaded = info["downloaded"]

            if downloaded:
                chunk_df = pd.read_json(os.path.join(save_dir, name), lines=True)
                df = pd.concat([df, chunk_df], axis=0)

        # Remove duplicates based on timestamp and title
        df = df.drop_duplicates(subset=["timestamp", "title"], keep="first")
        df = self._format_dataframe(df)

        if self.fetch_url:
            logger.info(f"| Fetching markdown content for {symbol} news items...")
            df = await self.download_markdown(df)
        else:
            df['content'] = ""
            df['markdown_path'] = ""

        df.to_json(os.path.join(self.exp_path, "{}.jsonl".format(symbol)), orient="records", lines=True)
        logger.info(f"| All data for {symbol} downloaded and saved to {self.exp_path}/{symbol}.jsonl")

class FMPSymbolInfoDownloader(AbstractDownloader):
    def __init__(self,
                 api_key: Optional[str] = None,
                 save_name: Optional[str] = "full",
                 exp_path: Optional[str] = None,
                 max_concurrent: Optional[int] = 10,
                 **kwargs):
        super().__init__()
        self.api_key = api_key

        if self.api_key is None:
            self.api_key = os.getenv("FMP_API_KEY")

        self.save_name = save_name

        self.list_url = "https://financialmodelingprep.com/stable/stock-list?apikey={}"
        self.info_url = "https://financialmodelingprep.com/stable/profile?symbol={}&apikey={}"
        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)

        self.max_concurrent = max_concurrent

    def _check_download(self, tasks):
        """
        Check if the symbol info file already exists.
        :return: True if the file exists, False otherwise.
        """
        os.makedirs(os.path.join(self.exp_path, "infos"), exist_ok=True)

        download_infos = []
        for task in tasks:
            symbol = task["symbol"]
            file_path = os.path.join(self.exp_path, "infos", "{}.json".format(symbol))
            if os.path.exists(file_path):
                item = {
                    "symbol": symbol,
                    "downloaded": True,
                    "file_path": file_path
                }
            else:
                item = {
                    "symbol": symbol,
                    "downloaded": False,
                    "file_path": file_path
                }
            download_infos.append(item)

        downloaded_items_num = len([info for info in download_infos if info["downloaded"]])
        total_items_num = len(download_infos)
        logger.info(f"| Symbol Info Downloaded / Total: [{downloaded_items_num} / {total_items_num}]")
        return download_infos

    async def run_task(self, task: Any):
        symbol = task["symbol"]
        symbol_info = await get_jsonparsed_data(self.info_url.format(symbol, self.api_key))
        logger.info(f"| Downloaded symbol info for {symbol}")

        if symbol_info:
            symbol_info = symbol_info[0]
            with open(os.path.join(self.exp_path, "infos", "{}.json".format(symbol)), "w") as op:
                json.dump(symbol_info, op, indent=4)
        else:
            symbol_info = None
            logger.error(f"| Failed to download symbol info for {symbol}. No data found.")
        return symbol_info

    async def run(self):
        """
        Download the stock list from Financial Modeling Prep and save it to the specified path.
        Returns:
        """
        infos = await get_jsonparsed_data(self.list_url.format(self.api_key))

        if not infos:
            logger.error("No symbols found in the stock list.")
            return

        tasks = self._check_download(infos)
        current_tasks = [task for task in tasks if not task["downloaded"]]

        symbol_info_list = []
        for i in range(0, len(current_tasks), self.max_concurrent):
            batch = current_tasks[i:min(i + self.max_concurrent, len(current_tasks))]
            results = await asyncio.gather(*[self.run_task(task) for task in batch])
            symbol_info_list.extend(results)

        download_infos = self._check_download(infos)
        download_infos = [info for info in download_infos if info["downloaded"]]
        sorted_symbols = sorted([info["symbol"] for info in download_infos])
        logger.info(f"| All symbols downloaded: [{len(sorted_symbols)}/ {len(download_infos)}]")

        res_infos = {}
        for symbol in sorted_symbols:
            file_path = os.path.join(self.exp_path, "infos", "{}.json".format(symbol))
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    res_infos[symbol] = json.load(f)
            else:
                logger.warning(f"| Symbol info for {symbol} not found in {file_path}.")

        with open(os.path.join(self.exp_path, "{}.json".format(self.save_name)), "w") as op:
            json.dump(res_infos, op, indent=4)