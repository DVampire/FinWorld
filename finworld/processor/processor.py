import os
import json
from typing import Any, Optional
import asyncio

from finworld.registry import PROCESSOR
from finworld.processor.custom import AbstractProcessor
from finworld.processor.fmp import (FMPPriceProcessor,
                                    FMPNewsProcessor,
                                    FMPFeatureProcessor)
from finworld.processor.alpaca import (AlpacaNewsProcessor,
                                       AlpacaPriceProcessor,
                                       AlpacaFeatureProcessor)
from finworld.processor.akshare import (
    AkSharePriceProcessor,
    AkShareNewsProcessor,
    AkShareFeatureProcessor
)
from finworld.processor.tushare import (
    TuSharePriceProcessor,
    TuShareNewsProcessor,
    TuShareFeatureProcessor
)
from finworld.utils import assemble_project_path, get_tag_name, push_to_hub_folder
from finworld.config import config
from finworld.log import logger

@PROCESSOR.register_module(force=True)
class Processor(AbstractProcessor):
    def __init__(self,
                 assets_name: Optional[str] = None,
                 source: Optional[str] = "fmp",
                 data_path: Optional[str] = None,
                 data_type: Optional[str] = None,
                 assets_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 level: Optional[str] = None,
                 format: Optional[str] = None,
                 max_concurrent: Optional[int] = None,
                 feature_type: Optional[str] = None,
                 **kwargs
                 ):
        super().__init__()

        self.assets_name = assets_name
        self.source = source
        self.data_path = assemble_project_path(data_path)
        self.data_type = data_type
        self.assets_path = assemble_project_path(assets_path)
        self.start_date = start_date
        self.end_date = end_date
        self.level = level
        self.format = format

        self.assets_info, self.symbols = self._load_assets()

        assert len(self.symbols) > 0, "No symbols to download"
        self.max_concurrent = max_concurrent
        self.feature_type = feature_type

        self.exp_path = config.exp_path
        os.makedirs(self.exp_path, exist_ok=True)

        self.processor_map = {
            "fmp": {
                "price": FMPPriceProcessor,
                "news": FMPNewsProcessor,
                "feature": FMPFeatureProcessor,
            },
            "alpaca": {
                "price": AlpacaPriceProcessor,
                "news": AlpacaNewsProcessor,
                "feature": AlpacaFeatureProcessor,
            },
            "akshare": {
                "price": AkSharePriceProcessor,
                "news": AkShareNewsProcessor,
                "feature": AkShareFeatureProcessor,
            },
            "tushare": {
                "price": TuSharePriceProcessor,
                "news": TuShareNewsProcessor,
                "feature": TuShareFeatureProcessor,
            }
        }

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
        return await task.run()

    async def _process(self, save_dir: str):
        """
        Process price data from FMP API.
        :return:
        """
        info = {
            "source": self.source,
            "data_type": self.data_type,
        }
        asset_info = []
        os.makedirs(save_dir, exist_ok=True)

        tasks = []
        for symbol in self.symbols:
            symbol_info = self._get_sybmol_info(symbol)

            assert self.source in self.processor_map, f"Source {self.source} not supported"
            source_processors = self.processor_map[self.source]
            assert self.data_type in source_processors, f"Data type {self.data_type} not supported for source {self.source}"
            class_ = source_processors[self.data_type]

            processor = class_(
                data_path=self.data_path,
                start_date=self.start_date,
                end_date=self.end_date,
                level=self.level,
                format=self.format,
                max_concurrent=self.max_concurrent,
                symbol_info=symbol_info,
                exp_path=save_dir,
                feature_type=self.feature_type
            )
            tasks.append(processor)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            batch_info = await asyncio.gather(*[self.run_task(task) for task in batch])
            asset_info.extend(batch_info)

        dates = {
            item["symbol"]: {
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date")
            } for item in asset_info
        }
        info["dates"] = dates
        info["names"] = asset_info[0].get("names", [])

        return info

    async def run(self):
        logger.info(f"| Processing {self.data_type} data from {self.source}...")
        tag_name = get_tag_name(assets_name=self.assets_name,
                                source=self.source,
                                data_type=self.data_type,
                                level=self.level)
        save_dir = os.path.join(self.exp_path, tag_name)
        os.makedirs(save_dir, exist_ok=True)
        info = await self._process(save_dir = save_dir)
        info["tag"] = tag_name

        return info

@PROCESSOR.register_module(force=True)
class AggProcessor(AbstractProcessor):
    def __init__(self,
                 procs_config: Optional[Any] = None,
                 assets_path: Optional[int] = None,
                 max_concurrent: Optional[int] = None,
                 repo_id: Optional[str] = None,
                 repo_type: Optional[str] = None,
                 ):
        super().__init__()

        self.procs_config = procs_config
        self.assets_path = assemble_project_path(assets_path)

        self.assets_info, self.symbols = self._load_assets()

        assert len(self.symbols) > 0, "No symbols to process"
        self.max_concurrent = max_concurrent
        self.repo_id = repo_id
        self.repo_type = repo_type

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
        return await task.run()

    async def run(self):
        """ Run the aggregation processor.
        """
        if self.procs_config is None:
            raise ValueError("No processors configured for aggregation")

        info = {
            "symbols": self.symbols,
            "symbols_info": self.assets_info,
        }
        data_info = []
        tasks = []
        for proc_config in self.procs_config:

            configs = {
                "type": proc_config.get("type", None),
                "assets_name": proc_config.get("assets_name", None),
                "source": proc_config.get("source", None),
                "data_path": proc_config.get("data_path", None),
                "data_type": proc_config.get("data_type", None),
                "assets_path": self.assets_path,
                "start_date": proc_config.get("start_date", None),
                "end_date": proc_config.get("end_date", None),
                "level": proc_config.get("level", None),
                "format": proc_config.get("format", None),
                "max_concurrent": self.max_concurrent,
                "feature_type": proc_config.get("feature_type", None),
            }

            processor = PROCESSOR.build(configs)

            tasks.append(processor)

        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:min(i + self.max_concurrent, len(tasks))]
            batch_info = await asyncio.gather(*[self.run_task(task) for task in batch])
            data_info.extend(batch_info)

        info['tags'] = [item['tag'] for item in data_info]
        info['data_info'] = {item['tag']: item for item in data_info}

        # Save the info
        info_path = os.path.join(self.exp_path, "meta_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

        logger.info(f"| Pushing processed data to Hugging Face Hub: {self.repo_id}...")
        push_to_hub_folder(
            hf_token=os.getenv("HF_API_KEY"),
            endpoint=os.getenv("HF_ENDPOINT"),
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            folder_path=self.exp_path,
        )