import enum

from finworld.registry import TASK

class TaskType(enum.Enum):
    """
    Enum for task types.
    """
    # Time Series
    FORECASTING = 'forecasting'
    IMPUTATION = 'imputation'
    OUTLIER = 'outlier'
    CLASSIFICATION = 'classification'

    # Pretrain
    PRETRAIN = 'pretrain'

    # Financial
    TRADING = 'trading'
    PORTFOLIO = 'portfolio'

    @classmethod
    def from_string(cls, task_type: str) -> 'TaskType':
        """
        Convert a string to a TaskType enum.
        """
        task_type = task_type.lower()

        if task_type == 'forecasting':
            return cls.FORECASTING
        elif task_type == 'imputation':
            return cls.IMPUTATION
        elif task_type == 'outlier':
            return cls.OUTLIER
        elif task_type == 'classification':
            return cls.CLASSIFICATION
        elif task_type == 'trading':
            return cls.TRADING
        elif task_type == 'portfolio':
            return cls.PORTFOLIO
        elif task_type == 'pretrain':
            return cls.PRETRAIN
        else:
            raise ValueError(f"Unknown task type: {task_type}")

@TASK.register_module(force=True)
class Task():
    def __init__(self,
                 trainer,
                 train: bool,
                 test: bool,
                 task_type: str
                 ):
        self.trainer = trainer
        self.train = train
        self.test = test
        self.task_type = TaskType.from_string(task_type)

    def run(self):
        if self.train:
            self.trainer.train()
        if self.test:
            self.trainer.test()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Task(trainer={self.trainer}, train={self.train}, test={self.test}, task={self.task_type.value})"

@TASK.register_module(force=True)
class AsyncTask():
    """
    Asynchronous task that can be run in a separate thread.
    """
    def __init__(self,
                 trainer,
                 train: bool,
                 test: bool,
                 task_type: str
                 ):
        self.trainer = trainer
        self.train = train
        self.test = test
        self.task_type = TaskType.from_string(task_type)

    async def run(self):
        if self.train:
            await self.trainer.train()
        if self.test:
            await self.trainer.test()
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"AsyncTask(trainer={self.trainer}, train={self.train}, test={self.test}, task={self.task_type.value})"
