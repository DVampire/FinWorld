import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from mmengine import DictAction
import torch
from pathlib import Path
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from finworld.config import config
from finworld.log import logger
from finworld.registry import DATASET
from finworld.registry import TRAINER
from finworld.registry import COLLATE_FN
from finworld.registry import MODEL
from finworld.registry import OPTIMIZER
from finworld.registry import SCHEDULER
from finworld.registry import LOSS
from finworld.registry import PLOT
from finworld.registry import TASK
from finworld.registry import METRIC
from finworld.utils import init_distributed_mode
from finworld.utils import to_torch_dtype, get_model_numel
from finworld.utils import get_world_size, get_rank

def parse_args():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--config", default=os.path.join(root, "configs", "time", "dj30_autoformer.py"), help="config file path")
    parser.add_argument("--train", action="store_true", help="whether to train the model")
    parser.set_defaults(train=True)
    parser.add_argument("--test", action="store_true", help="whether to test the model")
    parser.set_defaults(test=False)

    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", action="store_true")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()

    return args

def main():

    # Parse command line arguments
    args = parse_args()

    # Initialize the distributed mode
    init_distributed_mode(args)

    # Initialize the configuration
    config.init_config(args.config, args)

    # Initialize the logger
    logger.init_logger(config=config)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")

    # Set dtype
    dtype = to_torch_dtype(config.dtype)

    # Get the device
    device = torch.device(config.device)

    # Collate function
    collate_fn = COLLATE_FN.build(config.collate_fn)

    # Build the datasets
    dataset = DATASET.build(config.dataset)
    num_device = get_world_size()
    num_training_data = len(dataset)
    num_training_steps_per_epoch = int(np.floor(num_training_data / (config.batch_size * num_device)))
    num_training_steps = int(num_training_steps_per_epoch * config.num_training_epochs)
    num_training_warmup_steps = int(num_training_steps_per_epoch * config.num_training_warmup_epochs)
    config.merge_from_dict({
        "num_device": num_device,
        "num_training_data": num_training_data,
        "num_training_steps_per_epoch": num_training_steps_per_epoch,
        "num_training_steps": num_training_steps,
        "num_training_warmup_steps": num_training_warmup_steps,
    })

    # Build the model
    model = MODEL.build(config.model)
    logger.info(f"| Model: \n{model}")
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(f"| Model numel: {model_numel}, Trainable numel: {model_numel_trainable}")

    model.to(device=device, dtype=dtype)

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=True,
        )

    # Build optimizer
    optimizer_config = config.optimizer
    optimizer_config.update({
        "params": model.parameters(),
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    })
    optimizer = OPTIMIZER.build(optimizer_config)
    logger.info(f"| Optimizer: \n{optimizer}")

    # Build scheduler
    scheduler_config = config.scheduler
    scheduler_config.update({
        "optimizer": optimizer,
        "num_warmup_steps": num_training_warmup_steps,
        "num_training_steps": num_training_steps,
    })
    scheduler = SCHEDULER.build(scheduler_config)
    logger.info(f"| Scheduler: \n{scheduler}")

    # Build loss functions
    loss_configs = config.loss
    losses = {}
    for loss_name, loss_config in loss_configs.items():
        losses[loss_name] = LOSS.build(loss_config)
    logger.info(f"| Loss: \n{losses}")

    # Build metrics
    metric_configs = config.metric
    metrics = {}
    for metric_name, metric_config in metric_configs.items():
        metric_config.update(
            {
                "level": config.level,
                "symbol_info": dataset.assets_info[dataset.symbols[0]],
            }
        )
        metrics[metric_name] = METRIC.build(metric_config)
    logger.info(f"| Metrics: \n{metrics}")

    # Build plot
    if hasattr(config, "plot"):
        plot = PLOT.build(config.plot)
    else:
        plot = None

    trainer_config = config.trainer
    trainer_config.update(dict(
        config=config,
        model=model,
        collate_fn=collate_fn,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
        metrics=metrics,
        plot=plot,
        device=device,
        dtype=dtype,
    ))
    trainer = TRAINER.build(trainer_config)
    logger.info(f"| Trainer: \n{trainer}")

    # Build task
    task_config = config.task
    task_config.update({
        "trainer": trainer,
        "train": args.train,
        "test": args.test,
        "task_type": config.task_type
    })
    task = TASK.build(task_config)
    logger.info(f"| Task: \n{task}")

    # Run the task
    task.run()

if __name__ == '__main__':
    main()

    """
    CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29510 --nproc_per_node=2 scripts/time/train.py --config configs/time/dj30_autoformer.py
    """