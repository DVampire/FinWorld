from torch.optim import AdamW

from finworld.registry import OPTIMIZER

OPTIMIZER.register_module(name='AdamW', module=AdamW)
