from abc import ABC, abstractmethod
from typing import Any

class AbstractProcessor(ABC):

    def __init__(self, *args, **kwargs):
        """
        Initialize the processor class.
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

    @abstractmethod
    async def run(self, *args, **kwargs):
        """
        Run the processor.
        Args:
            *args:
            **kwargs:

        Returns:
        """
        pass