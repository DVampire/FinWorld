from abc import ABC, abstractmethod
from typing import Any

class AbstractDownloader(ABC):

    def __init__(self, *args, **kwargs):
        """
        Initialize the downloader class.
        Args:
            *args:
            **kwargs:
        """
        pass

    async def run_task(self, task: Any):
        """
        Run a single task.
        Args:
            task: The task to run.

        Returns:
            None
        """
        pass

    async def run(self, *args, **kwargs):
        """
        Run the downloader.
        Args:
            *args:
            **kwargs:

        Returns:
        """
        pass