import os
import torch
from typing import Any, Dict, List, Tuple, Union
from glob import glob
import json
import random

from finworld.registry import TRAINER
from finworld.utils import check_data
from finworld.utils import SmoothedValue
from finworld.utils import MetricLogger
from finworld.metric import MSE, RANKIC, RANKICIR

@TRAINER.register_module(force=True)
class FactorVAETrainer():
    def __init__(self,
                 *args,
                 batch_size,
                 start_epoch,
                 num_training_epochs,
                 num_checkpoint_del,
                 checkpoint_period,
                 vae,
                 vae_ema,
                 dit,
                 dit_ema,
                 diffusion,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 loss_funcs,
                 vae_optimizer,
                 dit_optimizer,
                 vae_scheduler,
                 dit_scheduler,
                 logger,
                 device,
                 dtype,
                 exp_path,
                 writer: Any = None,
                 wandb: Any = None,
                 num_plot_samples: int = 50,
                 plot: Any = None,
                 accelerator: Any = None,
                 print_freq: int = 20,
                 **kwargs):

        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.num_training_epochs = num_training_epochs
        self.num_valid_epochs = num_training_epochs
        self.num_testing_epochs = 1
        self.num_checkpoint_del = num_checkpoint_del
        self.checkpoint_period = checkpoint_period
        self.vae = vae
        self.vae_ema = vae_ema
        self.dit = dit
        self.dit_ema = dit_ema
        self.diffusion = diffusion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.vae_loss_fn = loss_funcs.get("vae_loss", None)
        self.price_cont_loss_fn = loss_funcs.get("price_cont_loss", None)
        self.vae_optimizer = vae_optimizer
        self.dit_optimizer = dit_optimizer
        self.vae_scheduler = vae_scheduler
        self.dit_scheduler = dit_scheduler
        self.logger = logger
        self.device = device
        self.dtype = dtype
        self.exp_path = exp_path
        self.writer = writer
        self.wandb = wandb
        self.num_plot_samples = num_plot_samples
        self.plot = plot
        self.accelerator = accelerator
        self.print_freq = print_freq

        self._init_params()

    def _init_params(self):

        self.logger.info("| Init parameters for VAE trainer...")

        torch.set_default_dtype(self.dtype)

        self.is_main_process = self.accelerator.is_local_main_process

        self.model = self.accelerator.prepare(self.vae)
        self.model_ema = self.accelerator.prepare(self.vae_ema)

        if self.vae_loss_fn:
            self.vae_loss_fn = self.accelerator.prepare(self.vae_loss_fn)
        if self.price_cont_loss_fn:
            self.price_cont_loss_fn = self.accelerator.prepare(self.price_cont_loss_fn)

        self.optimizer = self.accelerator.prepare(self.vae_optimizer)
        self.scheduler = self.accelerator.prepare(self.vae_scheduler)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(self.valid_dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        torch.set_default_dtype(self.dtype)

        self.checkpoint_path = os.path.join(self.exp_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.plot_path = os.path.join(self.exp_path, "plot")
        os.makedirs(self.plot_path, exist_ok=True)

        self.start_epoch = self.load_checkpoint() + 1

        self.check_batch_info_flag = True

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

    def save_checkpoint(self, epoch: int, if_best: bool = False):
        if not self.accelerator.is_local_main_process:
            return  # Only save checkpoint on the main process

        if if_best:
            checkpoint_file = os.path.join(self.checkpoint_path, f"best.pth")
        else:
            checkpoint_file = os.path.join(self.checkpoint_path, "checkpoint_{:06d}.pth".format(epoch))

        # Save model, optimizer, and scheduler states
        state = {
            'epoch': epoch,
            'model_state': self.accelerator.unwrap_model(self.model).state_dict(),
            'model_ema_state': self.accelerator.unwrap_model(self.model_ema).state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        self.accelerator.save(state, checkpoint_file)

        # Manage saved checkpoints
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(checkpoint_files) > self.num_checkpoint_del:
            for checkpoint_file in checkpoint_files[:-self.num_checkpoint_del]:
                os.remove(checkpoint_file)
                self.logger.info(f"｜ Checkpoint deleted: {checkpoint_file}")

        self.logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, epoch: int = -1, if_best: bool = False):

        best_checkpoint_file = os.path.join(self.checkpoint_path, "best.pth")
        epoch_checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch:06d}.pth")

        latest_checkpoint_file = None
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        if not checkpoint_files:
            self.logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
        else:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_file = checkpoint_files[-1]
            latest_checkpoint_file = checkpoint_file

        if if_best and os.path.exists(best_checkpoint_file):
            checkpoint_file = best_checkpoint_file
            self.logger.info(f"| Load best checkpoint: {checkpoint_file}")
        elif epoch >= 0 and os.path.exists(epoch_checkpoint_file):
            checkpoint_file = epoch_checkpoint_file
            self.logger.info(f"| Load epoch checkpoint: {checkpoint_file}")
        else:
            if latest_checkpoint_file:
                checkpoint_file = latest_checkpoint_file
                self.logger.info(f"| Load latest checkpoint: {checkpoint_file}")
            else:
                checkpoint_file = None
                self.logger.info(f"| Checkpoint not found.")

        if checkpoint_file is not None:

            state = torch.load(checkpoint_file, map_location=self.device)

            # Unwrap the model to load state dict into the underlying model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(state['model_state'])

            unwrapped_model_ema = self.accelerator.unwrap_model(self.model_ema)
            unwrapped_model_ema.load_state_dict(state['model_ema_state'])

            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])

            return state['epoch']
        else:
            return 0

    def __str__(self):
        return f"FactorVAETrainer(num_training_epochs={self.num_training_epochs}, start_epoch={self.start_epoch}, batch_size={self.batch_size})"

    def check_batch_info(self, batch: Dict):

        if self.check_batch_info_flag:
            asset = batch["asset"]
            self.logger.info(f"| Asset: {check_data(asset)}")

            for key in batch:
                if key not in ["asset"]:
                    data = batch[key]
                    log_str = f"| {key}: "
                    for key, value in data.items():
                        log_str += f"\n {key}: {check_data(value)}"
                    self.logger.info(log_str)

            self.check_batch_info_flag = False

    def run_step(self,
                 epoch,
                 if_use_writer = True,
                 if_use_wandb = True,
                 if_plot = False,
                 mode = "train"
                 ):

        if_train = mode == "train"

        records = dict()
        metric_logger = MetricLogger(delimiter="  ")

        if if_train:
            self.model.train(True)
        else:
            self.model.eval()

        if mode == "train":
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

            header = f"| Train Epoch: [{epoch}/{self.num_training_epochs}]"
            global_step = self.global_train_step
            dataloader = self.train_dataloader
        elif mode == "valid":
            header = f"| Valid Epoch: [{epoch}/{self.num_valid_epochs}]"
            global_step = self.global_valid_step
            dataloader = self.valid_dataloader
        else:
            header = f"| Test Epoch: [{epoch}/{self.num_testing_epochs}]"
            global_step = self.global_test_step
            dataloader = self.test_dataloader

        rankics = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       self.logger,
                                                                       self.print_freq,
                                                                       header)):

            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            self.check_batch_info(batch)

            asset = batch["asset"] # (N, T)
            history = batch["history"]

            start_timestamp = history["start_timestamp"] # (N,)
            end_timestamp = history["end_timestamp"] # (N,)
            start_index = history["start_index"] # (N,)
            end_index = history["end_index"] # (N,)
            features = history["features"] # (N, T, S, F)
            labels = history["labels"] # (N, T, S, 2)
            prices = history["prices"] # (N, T, S, 5)
            timestamps = history["timestamps"] # (N, T, S)
            prices_mean = history["prices_mean"] # (N, T, S, 5)
            prices_std = history["prices_std"] # (N, T, S, 5)
            text = history["text"] # (N, T)

            features = features.to(self.device, self.dtype)

            if len(features.shape) == 4:
                features = features.unsqueeze(1) # (N, C, T, S, F)
            if len(labels.shape) == 4:
                labels = labels.unsqueeze(1) # (N, C, T, S, 2)

            # The next day returns of the last day
            labels = labels[:, :, -1:, :, 0] # (N, C, T, S)

            # Forward
            if if_train:
                output = self.model(features, labels, training = True)
            else:
                with torch.no_grad():
                    output = self.model(features, labels, training = False)

            pred_returns = output["sample"]
            target_returns = labels.squeeze(1).squeeze(1)
            mu_post = output["mu_post"]
            sigma_post = output["sigma_post"]
            mu_pred = output["mu_pred"]
            sigma_pred = output["sigma_pred"]

            if self.vae_loss_fn:
                loss_dict = self.vae_loss_fn(sample=pred_returns,
                                             target_sample=target_returns,
                                             mu_post=mu_post,
                                             sigma_post=sigma_post,
                                             mu_pred=mu_pred,
                                             sigma_pred=sigma_pred)

                weighted_nll_loss = loss_dict["weighted_nll_loss"]
                weighted_kl_loss = loss_dict["weighted_kl_loss"]

                records.update({
                    "weighted_nll_loss": self.accelerator.gather(weighted_nll_loss).mean().item(),
                    "weighted_kl_loss": self.accelerator.gather(weighted_kl_loss).mean().item()
                })

                loss += weighted_nll_loss + weighted_kl_loss

            records.update({
                "loss": self.accelerator.gather(loss).mean().item()
            })

            pred_returns = torch.where(torch.isnan(pred_returns), torch.full_like(pred_returns, 0), pred_returns)

            pred_returns = pred_returns.detach().cpu().numpy()
            target_returns = target_returns.detach().cpu().numpy()
            mse = MSE(target_returns, pred_returns)
            records.update({
                "mse": mse
            })

            rankic = RankIC(pred_returns, target_returns)
            records.update({
                "rankic": rankic,
            })
            rankics.append(rankic)

            rankicir = RankICIR(rankics)
            records.update({
                "rankicir": rankicir,
            })

            if if_train:
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                records.update({
                    "lr": lr
                })

            global_step += 1

            prefix = mode
            if global_step % self.print_freq == 0:

                wandb_dict = {}
                # For records
                for key, value in records.items():
                    if if_use_writer and self.writer:
                        self.writer.log_scalar(f"{prefix}/{key}", value, global_step)
                    if if_use_wandb and self.wandb:
                        wandb_dict[f"{prefix}/{key}"] = value

                self.wandb.log(wandb_dict)

            metric_logger.update(**records)
            metric_logger.synchronize_between_processes()

        if if_use_writer and self.is_main_process:
            self.writer.flush()

        if mode == "train":
            self.global_train_step = global_step
            log_str = "| Train averaged stats: "
        elif mode == "valid":
            self.global_valid_step = global_step
            log_str = "| Valid averaged stats: "
        else:
            self.global_test_step = global_step
            log_str = "| Test averaged stats: "

        for name, meter in metric_logger.meters.items():
            log_str += f"- {name}: {meter}"
        self.logger.info(log_str)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self):

        self.logger.info("| Start training and evaluating VAE...")

        min_metric = float("inf")

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):
            train_stats = self.run_step(epoch, mode="train")
            valid_stats = self.run_step(epoch, mode="valid")

            metric = valid_stats["mse"]

            log_stats = {"epoch": epoch}
            log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
            log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_log.txt"), "a",) as f:
                    f.write(json.dumps(log_stats) + "\n")

            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(epoch)

            if metric < min_metric:
                min_metric = metric
                self.save_checkpoint(epoch, if_best=True)

    def test(self):
        self.logger.info("| Start testing VAE...")

        best_checkpoint_path = os.path.join(self.checkpoint_path, "best.pth")
        if os.path.exists(best_checkpoint_path):
            epoch = self.load_checkpoint(if_best=True)
        else:
            self.logger.info("| Best checkpoint not found. Load the last checkpoint.")
            epoch = self.load_checkpoint()

        log_stats = {"epoch": epoch}

        train_stats = self.run_step(epoch,
                                   mode="train",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        valid_stats = self.run_step(epoch,
                                   mode="valid",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        test_stats = self.run_step(epoch,
                                   mode="test",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=False)

        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
        log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "test_log.txt"), "w",) as f:
                f.write(json.dumps(log_stats) + "\n")

        self.logger.info("| Test finished.")