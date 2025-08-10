import numpy as np
import torch
from typing import Dict, Any
from finworld.registry import COLLATE_FN

@COLLATE_FN.register_module(force=True)
class SingleAssetPriceTextCollateFn():
    def __init__(self):
        super().__init__()

    def __call__(self, batch):

        batch_size = len(batch)

        collect_batch: Dict[str, Any] = {}

        for i in range(batch_size):
            # asset
            asset = batch[i]["asset"]
            collect_batch.setdefault("asset", []).append(asset)

            batch_data_names = [name for name in batch[i].keys() if name not in ["asset"]] # history, future
            for batch_data_name in batch_data_names:
                collect_batch.setdefault(batch_data_name, {})

                batch_data = batch[i][batch_data_name]

                start_timestamp = batch_data["start_timestamp"]
                end_timestamp = batch_data["end_timestamp"]
                start_index = batch_data["start_index"]
                end_index = batch_data["end_index"]
                prices = batch_data["prices"]
                labels = batch_data["labels"]
                original_prices = batch_data["original_prices"]
                timestamps = batch_data["timestamps"]
                prices_mean = batch_data["prices_mean"]
                prices_std = batch_data["prices_std"]

                collect_batch[batch_data_name].setdefault("start_timestamp", []).append(start_timestamp)
                collect_batch[batch_data_name].setdefault("end_timestamp", []).append(end_timestamp)
                collect_batch[batch_data_name].setdefault("start_index", []).append(start_index)
                collect_batch[batch_data_name].setdefault("end_index", []).append(end_index)
                collect_batch[batch_data_name].setdefault("prices", []).append(prices)
                collect_batch[batch_data_name].setdefault("labels", []).append(labels)
                collect_batch[batch_data_name].setdefault("original_prices", []).append(original_prices)
                collect_batch[batch_data_name].setdefault("timestamps", []).append(timestamps)
                collect_batch[batch_data_name].setdefault("prices_mean", []).append(prices_mean)
                collect_batch[batch_data_name].setdefault("prices_std", []).append(prices_std)

                if "features" in batch_data and batch_data["features"] is not None:
                    features = batch_data["features"]
                    collect_batch[batch_data_name].setdefault("features", []).append(features)
                if "times" in batch_data and batch_data["times"] is not None:
                    times = batch_data["times"]
                    collect_batch[batch_data_name].setdefault("times", []).append(times)
                if "news" in batch_data and batch_data["news"] is not None:
                    news = batch_data["news"]
                    collect_batch[batch_data_name].setdefault("news", []).append(news)

        for batch_data_name in collect_batch:
            if batch_data_name not in ["asset"]:
                collect_batch[batch_data_name]["start_timestamp"] = collect_batch[batch_data_name]["start_timestamp"] # Timestamp
                collect_batch[batch_data_name]["end_timestamp"] = collect_batch[batch_data_name]["end_timestamp"] # Timestamp
                collect_batch[batch_data_name]["start_index"] = torch.from_numpy(np.asarray(collect_batch[batch_data_name]["start_index"], dtype=np.int64))
                collect_batch[batch_data_name]["end_index"] = torch.from_numpy(np.asarray(collect_batch[batch_data_name]["end_index"], dtype=np.int64))
                collect_batch[batch_data_name]["prices"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices"]))
                collect_batch[batch_data_name]["labels"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["labels"]))
                collect_batch[batch_data_name]["original_prices"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["original_prices"]))
                collect_batch[batch_data_name]["timestamps"] = collect_batch[batch_data_name]["timestamps"] # Timestamp list
                collect_batch[batch_data_name]["prices_mean"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices_mean"]))
                collect_batch[batch_data_name]["prices_std"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices_std"]))

                if "features" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["features"] is not None:
                    collect_batch[batch_data_name]["features"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["features"]))
                if "times" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["times"] is not None:
                    collect_batch[batch_data_name]["times"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["times"]))
                if "news" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["news"] is not None:
                    collect_batch[batch_data_name]["news"] = collect_batch[batch_data_name]["news"]

        return collect_batch

@COLLATE_FN.register_module(force=True)
class MultiAssetPriceTextCollateFn():
    def __init__(self):
        super().__init__()

    def __call__(self, batch):

        batch_size = len(batch)

        collect_batch: Dict[str, Any] = {}

        for i in range(batch_size):
            # asset
            assets = batch[i]["assets"]
            collect_batch.setdefault("assets", []).append(assets)

            batch_data_names = [name for name in batch[i].keys() if name not in ["assets"]] # history, future
            for batch_data_name in batch_data_names:
                collect_batch.setdefault(batch_data_name, {})

                batch_data = batch[i][batch_data_name]

                start_timestamp = batch_data["start_timestamp"]
                end_timestamp = batch_data["end_timestamp"]
                start_index = batch_data["start_index"]
                end_index = batch_data["end_index"]
                prices = batch_data["prices"].transpose(1, 0, 2)
                labels = batch_data["labels"].transpose(1, 0, 2)
                original_prices = batch_data["original_prices"].transpose(1, 0, 2)
                timestamps = batch_data["timestamps"]
                prices_mean = batch_data["prices_mean"].transpose(1, 0, 2)
                prices_std = batch_data["prices_std"].transpose(1, 0, 2)

                collect_batch[batch_data_name].setdefault("start_timestamp", []).append(start_timestamp)
                collect_batch[batch_data_name].setdefault("end_timestamp", []).append(end_timestamp)
                collect_batch[batch_data_name].setdefault("start_index", []).append(start_index)
                collect_batch[batch_data_name].setdefault("end_index", []).append(end_index)
                collect_batch[batch_data_name].setdefault("prices", []).append(prices)
                collect_batch[batch_data_name].setdefault("labels", []).append(labels)
                collect_batch[batch_data_name].setdefault("original_prices", []).append(original_prices)
                collect_batch[batch_data_name].setdefault("timestamps", []).append(timestamps)
                collect_batch[batch_data_name].setdefault("prices_mean", []).append(prices_mean)
                collect_batch[batch_data_name].setdefault("prices_std", []).append(prices_std)

                if "features" in batch_data and batch_data["features"] is not None:
                    features = batch_data["features"].transpose(1, 0, 2)
                    collect_batch[batch_data_name].setdefault("features", []).append(features)
                if "times" in batch_data and batch_data["times"] is not None:
                    times = batch_data["times"]
                    collect_batch[batch_data_name].setdefault("times", []).append(times)
                if "news" in batch_data and batch_data["news"] is not None:
                    news = batch_data["news"]
                    collect_batch[batch_data_name].setdefault("news", []).append(news)

        for batch_data_name in collect_batch:
            if batch_data_name not in ["assets"]:
                collect_batch[batch_data_name]["start_timestamp"] = collect_batch[batch_data_name]["start_timestamp"] # Timestamp
                collect_batch[batch_data_name]["end_timestamp"] = collect_batch[batch_data_name]["end_timestamp"] # Timestamp
                collect_batch[batch_data_name]["start_index"] = torch.from_numpy(np.asarray(collect_batch[batch_data_name]["start_index"], dtype=np.int64))
                collect_batch[batch_data_name]["end_index"] = torch.from_numpy(np.asarray(collect_batch[batch_data_name]["end_index"], dtype=np.int64))
                collect_batch[batch_data_name]["prices"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices"]))
                collect_batch[batch_data_name]["labels"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["labels"]))
                collect_batch[batch_data_name]["original_prices"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["original_prices"]))
                collect_batch[batch_data_name]["timestamps"] = collect_batch[batch_data_name]["timestamps"] # Timestamp list
                collect_batch[batch_data_name]["prices_mean"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices_mean"]))
                collect_batch[batch_data_name]["prices_std"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["prices_std"]))

                if "features" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["features"] is not None:
                    collect_batch[batch_data_name]["features"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["features"]))
                if "times" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["times"] is not None:
                    collect_batch[batch_data_name]["times"] = torch.from_numpy(np.array(collect_batch[batch_data_name]["times"]))
                if "news" in collect_batch[batch_data_name] and collect_batch[batch_data_name]["news"] is not None:
                    collect_batch[batch_data_name]["news"] = collect_batch[batch_data_name]["news"]

        return collect_batch