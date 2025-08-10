import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from mmengine import DictAction
from copy import deepcopy
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from finworld.config import config
from finworld.log import logger
from finworld.log import tensorboard_logger
from finworld.log import wandb_logger
from finworld.registry import DATASET
from finworld.registry import COLLATE_FN
from finworld.registry import TRAINER
from finworld.registry import MODEL
from finworld.registry import OPTIMIZER
from finworld.registry import SCHEDULER
from finworld.registry import LOSS_FUNC
from finworld.registry import PLOT
from finworld.registry import DATALOADER
from finworld.utils import to_torch_dtype
from finworld.utils import get_model_numel
from finworld.utils import requires_grad
from finworld.utils import record_model_param_shape

def parse_args():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--config", default=os.path.join(root, "configs", "vae", "transformer_vae"), help="config file path")
    parser.add_argument("--train", action="store_true", help="whether to train the model")
    parser.set_defaults(train=True)
    parser.add_argument("--test", action="store_true", help="whether to test the model")
    parser.set_defaults(test=False)

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

    # Initialize the configuration
    config.init_config(args.config, args)

    # Init Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=True if config.device == "cpu" else False, kwargs_handlers=[ddp_kwargs])

    # Initialize the logger
    logger.init_logger(log_path=config.log_path, accelerator=accelerator)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")
    if config.tensorboard_path:
        tensorboard_logger.init_logger(config.tensorboard_path, accelerator=accelerator)
        logger.info(f"| Tensorboard logger initialized at: {config.tensorboard_path}")
    if config.wandb_path:
        wandb_logger.init_logger(
            project=config.project,
            name=config.tag,
            config=config.to_dict(),
            dir=config.wandb_path,
            accelerator=accelerator,
        )
        logger.info(f"| WandB logger initialized at: {config.wandb_path}")

    # Set dtype
    dtype = to_torch_dtype(config.dtype)

    # Get the device
    device = accelerator.device

    # Init collate function
    collate_fn = COLLATE_FN.build(config.collate_fn)

    # Build the dataset and dataloader
    train_dataset = DATASET.build(config.train_dataset)
    logger.info(f"| Train dataset: \n{train_dataset}")
    train_dataloader_config = deepcopy(config.train_dataloader)
    train_dataloader_config.update({
        "accelerator": accelerator,
        "collate_fn": collate_fn,
        "dataset": train_dataset,
    })
    train_dataloader = DATALOADER.build(train_dataloader_config)

    num_device = torch.cuda.device_count()
    num_training_data = len(train_dataset)
    num_training_steps_per_epoch = int(np.floor(num_training_data / (config.batch_size * num_device)))
    num_training_steps = int(num_training_steps_per_epoch * config.num_training_epochs)
    num_training_warmup_steps = int(num_training_steps_per_epoch * config.num_training_warmup_epochs)
    config.merge_from_dict({
        "num_training_data": num_training_data,
        "num_training_steps_per_epoch": num_training_steps_per_epoch,
        "num_training_steps": num_training_steps,
        "num_training_warmup_steps": num_training_warmup_steps,
    })

    valid_dataset = DATASET.build(config.valid_dataset)
    logger.info(f"| Valid dataset: \n{valid_dataset}")
    valid_dataloader_config = deepcopy(config.valid_dataloader)
    valid_dataloader_config.update({
        "accelerator": accelerator,
        "collate_fn": collate_fn,
        "dataset": valid_dataset,
    })
    valid_dataloader = DATALOADER.build(valid_dataloader_config)

    test_dataset = DATASET.build(config.test_dataset)
    logger.info(f"| Test dataset: \n{test_dataset}")
    test_dataloader_config = deepcopy(config.test_dataloader)
    test_dataloader_config.update({
        "accelerator": accelerator,
        "collate_fn": collate_fn,
        "dataset": test_dataset,
    })
    test_dataloader = DATALOADER.build(test_dataloader_config)

    # Build the model
    vae = MODEL.build(config.vae)
    logger.info(f"| VAE: \n{vae}")
    vae_model_numel, vae_model_numel_trainable = get_model_numel(vae)
    logger.info(f"| VAE model numel: {vae_model_numel}, trainable: {vae_model_numel_trainable}")

    # Build VAE ema
    vae_state_dict = vae.state_dict()
    vae_ema = MODEL.build(config.vae)
    vae_ema.load_state_dict(vae_state_dict)
    vae_ema = vae_ema.to(device, dtype)
    requires_grad(vae_ema, True)
    vae_ema_shape_dict = record_model_param_shape(vae_ema)
    logger.info(f"| VAE EMA: \n{vae_ema}")
    logger.info("| VAE EMA shape: \n{}".format("\n".join([f"{k}: {v}" for k, v in vae_ema_shape_dict.items()])))

    # Move VAE to device and dtype
    vae = vae.to(device, dtype)

    # Build loss functions
    loss_funcs_config = config.loss_funcs_config
    loss_funcs = {}
    for loss_func_name, loss_func_config in loss_funcs_config.items():
        loss_funcs[loss_func_name] = LOSS_FUNC.build(loss_func_config).to(device, dtype)
        logger.info(f"| {loss_func_name} loss function: \n{loss_funcs[loss_func_name]}")

    # Build optimizer
    vae_params_groups = vae.parameters()
    vae_params_groups = filter(lambda p: p.requires_grad, vae_params_groups)
    vae_optimizer_config = deepcopy(config.vae_optimizer)
    vae_optimizer_config["params"] = vae_params_groups
    vae_optimizer = OPTIMIZER.build(vae_optimizer_config)
    logger.info(f"| VAE optimizer: \n{vae_optimizer}")

    # Build lr scheduler
    vae_scheduler_config = deepcopy(config.vae_scheduler)
    vae_scheduler_config.update({
        "num_training_steps": config.num_training_steps,
        "num_warmup_steps": config.num_training_warmup_steps,
        "optimizer": vae_optimizer
    })
    vae_scheduler = SCHEDULER.build(vae_scheduler_config)
    logger.info(f"| VAE scheduler: \n{vae_scheduler}")

    # Build plot
    if hasattr(config, "plot"):
        plot = PLOT.build(config.plot)
    else:
        plot = None

    # Build trainer
    trainer_config = deepcopy(config.trainer)
    trainer_config.update({
        "config": config,
        "vae": vae,
        "vae_ema": vae_ema,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "loss_funcs": loss_funcs,
        "vae_optimizer": vae_optimizer,
        "vae_scheduler": vae_scheduler,
        "device": device,
        "dtype": dtype,
        "writer": tensorboard_logger,
        "wandb": wandb_logger,
        "plot": plot,
        "accelerator": accelerator,
    })
    trainer = TRAINER.build(trainer_config)
    logger.info(f"| Trainer: \n{trainer}")

    # Start training + evaluation
    logger.info(f"| Train: {args.train}")
    if args.train:
        trainer.train()

    # Start testing
    logger.info(f"| Test: {args.test}")
    if args.test:
        trainer.test()


if __name__ == '__main__':
    main()

    """
    accelerate launch --main_process_port 29507 scripts/vae/train.py --config configs/vae/transformer_vae.py
    """