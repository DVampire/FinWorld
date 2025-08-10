import torch
from typing import Any
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from finworld.utils import get_world_size
from finworld.utils import get_rank
from finworld.registry import DATALOADER


@DATALOADER.register_module(force=True)
class DataLoader(torch.utils.data.DataLoader):
    """
    A wrapper for `torch.utils.data.DataLoader` to support both distributed and non-distributed training.
    Automatically handles sampler logic for train/eval mode and distributed settings.
    """

    def __init__(
        self,
        dataset: Dataset,
        collate_fn: Any,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        distributed: bool = False,
        train: bool = True,
        **kwargs,
    ):
        if train:
            if distributed:
                num_device = get_world_size()
                rank = get_rank()
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=num_device,
                    rank=rank,
                    shuffle=True  # always shuffle for train
                )
            else:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
