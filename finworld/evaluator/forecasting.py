import numpy as np
import os
from typing import Any
import torch

from finworld.registry import EVALUATOR
from finworld.log import logger

@EVALUATOR.register_module(force=True)
class ForecastingEvaluator():
    def __init__(self,
                 config,
                 accelerator,
                 model,
                 dataloader,
                 device,
                 dtype,
                 ):
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.dtype = dtype

        self._init_params()

    def _init_params(self):

        self.model = self.accelerator.prepare(self.model)
        self.dataloader = self.accelerator.prepare(self.dataloader)

        torch.set_default_dtype(self.dtype)
