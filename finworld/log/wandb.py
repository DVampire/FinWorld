import os
import wandb
from dotenv import load_dotenv
from typing import Optional, Any, Dict

load_dotenv(verbose=True)

from finworld.utils import is_main_process
from finworld.utils import Singleton


__all__ = [
    'WandbLogger',
    'wandb_logger'
]

class WandbLogger(metaclass=Singleton):
    def __init__(self):
        self.is_main_process = True

    def init_logger(self, config):

        self.is_main_process = is_main_process()

        project = config['project']
        name = config['name']
        dir = config['logging_dir']

        if self.is_main_process:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project=project, name=name, dir=dir)

    def log(self, metric: Dict[str, Any]):
        if not self.is_main_process:
            return
        wandb.log(metric)

    def finish(self):
        if not self.is_main_process:
            return
        wandb.finish()

wandb_logger = WandbLogger()