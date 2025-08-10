import os

import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Union
from glob import glob
import json
import random
from typing import NamedTuple
from datetime import datetime
import pandas as pd

from finworld.registry import TRAINER
from finworld.utils import check_data
from finworld.utils import SmoothedValue
from finworld.utils import MetricLogger
from finworld.metric import MSE
from finworld.models.embed import patchify, get_patch_info
from finworld.log import logger


@TRAINER.register_module(force=True)
class VAETrainer():
    def __init__(self,
                 *args,
                 config,
                 vae,
                 vae_ema,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 loss_funcs,
                 vae_optimizer,
                 vae_scheduler,
                 device,
                 dtype,
                 writer: Any = None,
                 wandb: Any = None,
                 plot: Any = None,
                 accelerator: Any = None,
                 **kwargs):
        self.config = config
        self.vae = vae
        self.vae_ema = vae_ema
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.vae_loss_fn = loss_funcs.get("vae_loss", None)
        self.price_cont_loss_fn = loss_funcs.get("price_cont_loss", None)

        self.vae_optimizer = vae_optimizer
        self.vae_scheduler = vae_scheduler
        self.device = device
        self.dtype = dtype
        self.writer = writer
        self.wandb = wandb
        self.plot = plot
        self.accelerator = accelerator

        self.plot_path = self.config.plot_path
        self.checkpoint_path = self.config.checkpoint_path
        self.check_batch_info_flag = True
        self.num_training_epochs = self.config.num_training_epochs
        self.num_valid_epochs = self.config.num_training_epochs
        self.num_testing_epochs = 1
        self.checkpoint_period = self.config.checkpoint_period
        self.num_checkpoint_del = self.config.num_checkpoint_del
        self.input_channel = self.config.input_channel
        self.batch_size = self.config.batch_size
        self.patch_size = self.config.patch_size
        self.if_mask = self.config.if_mask
        self.print_freq = self.config.print_freq
        self.exp_path = self.config.exp_path
        self.num_plot_samples = self.config.num_plot_samples
        self.num_plot_samples_per_batch = self.config.num_plot_samples_per_batch
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"

        self.global_train_step = 0
        self.global_valid_step = 0
        self.global_test_step = 0

        self._init_params()

        self.start_epoch = self.load_checkpoint() + 1

    def _init_params(self):
        logger.info("| Init parameters for VAE trainer...")

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
                logger.info(f"｜ Checkpoint deleted: {checkpoint_file}")

        logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, epoch: int = -1, if_best: bool = False):

        best_checkpoint_file = os.path.join(self.checkpoint_path, "best.pth")
        epoch_checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch:06d}.pth")

        latest_checkpoint_file = None
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        if not checkpoint_files:
            logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
        else:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_file = checkpoint_files[-1]
            latest_checkpoint_file = checkpoint_file

        if if_best and os.path.exists(best_checkpoint_file):
            checkpoint_file = best_checkpoint_file
            logger.info(f"| Load best checkpoint: {checkpoint_file}")
        elif epoch >= 0 and os.path.exists(epoch_checkpoint_file):
            checkpoint_file = epoch_checkpoint_file
            logger.info(f"| Load epoch checkpoint: {checkpoint_file}")
        else:
            if latest_checkpoint_file:
                checkpoint_file = latest_checkpoint_file
                logger.info(f"| Load latest checkpoint: {checkpoint_file}")
            else:
                checkpoint_file = None
                logger.info(f"| Checkpoint not found.")

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
        return f"VAETrainer(num_training_epochs={self.num_training_epochs}, start_epoch={self.start_epoch}, batch_size={self.batch_size})"

    def check_batch_info(self, batch: Dict):

        if self.check_batch_info_flag:
            asset = batch["asset"]
            logger.info(f"| Asset: {check_data(asset)}")

            for key in batch:
                if key not in ["asset"]:
                    data = batch[key]
                    log_str = f"| {key}: "
                    for key, value in data.items():
                        log_str += f"\n {key}: {check_data(value)}"
                    logger.info(log_str)

            self.check_batch_info_flag = False

    def _sample_plot(self,
                     start_timestamps: List,
                     end_timestamps: List,
                     timestamps: List,
                     restored_target_prices: np.ndarray,
                     restored_pred_prices: np.ndarray,
                     volumes: np.ndarray,
                     save_path: str = None):
        """
        Plot random samples from a batch of price sequences (no asset dimension).
        """
        batch_size = restored_target_prices.shape[0]
        assert batch_size == len(timestamps) == len(restored_target_prices) == len(restored_pred_prices), \
            "Length mismatch among timestamps and price arrays."

        sample_num = min(batch_size,
                         self.config.num_plot_samples_per_batch) if self.config.num_plot_samples_per_batch is not None else batch_size
        sample_indices = random.sample(list(range(batch_size)), sample_num)

        for sample_index in sample_indices:
            # Convert timestamps to readable strings
            start_ts = start_timestamps[sample_index].strftime(self.timestamp_format)
            end_ts = end_timestamps[sample_index].strftime(self.timestamp_format)

            ts = [t.strftime(self.timestamp_format) for t in timestamps[sample_index]]


            sample_price = restored_target_prices[sample_index]  # shape (T, 4)
            sample_pred_price = restored_pred_prices[sample_index]  # shape (T, 4)
            sample_volume = volumes[sample_index]  # shape (T,)

            prices = pd.DataFrame({
                "Date": ts,
                "Close": sample_price[:, 0],
                "High": sample_price[:, 1],
                "Low": sample_price[:, 2],
                "Open": sample_price[:, 3],
                "Volume": sample_volume,
            })
            pred_prices = pd.DataFrame({
                "Date": ts,
                "Close": sample_pred_price[:, 0],
                "High": sample_pred_price[:, 1],
                "Low": sample_pred_price[:, 2],
                "Open": sample_pred_price[:, 3],
                "Volume": sample_volume,
            })
            prices['Date'] = pd.to_datetime(prices['Date'])
            pred_prices['Date'] = pd.to_datetime(pred_prices['Date'])
            prices.set_index('Date', inplace=True)
            pred_prices.set_index('Date', inplace=True)

            prices_title = f"GT Prices from {start_ts} to {end_ts}"
            pred_prices_title = f"Predicted Prices from {start_ts} to {end_ts}"

            self.plot(
                {
                    prices_title: prices,
                    pred_prices_title: pred_prices,
                },
                show_volume=True,
                show_macd=True,
                savefig=save_path,
            )

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

        patch_size = self.patch_size
        if_mask = self.if_mask

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

        sample_batchs = []
        if if_plot:
            num_plot_sample_batch = int(self.num_plot_samples // self.num_plot_samples_per_batch)
            sample_batchs = random.sample(range(len(dataloader)), num_plot_sample_batch)

        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader,
                                                                       logger,
                                                                       self.print_freq,
                                                                       header)):

            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            # Load batch data
            self.check_batch_info(batch)

            history = batch["history"]

            start_timestamp = history["start_timestamp"] # (N,)
            end_timestamp = history["end_timestamp"] # (N,)
            start_index = history["start_index"] # (N,)
            end_index = history["end_index"] # (N,)
            prices = history["prices"]  # (N, T, 5), 'close', 'high', 'low', 'open', 'volume'
            features = history["features"] # (N, T, F)
            labels = history["labels"] # (N, T, 2)
            times = history["times"] # (N, T, 4), 'day', 'month', 'weekday', 'year'
            original_prices = history["original_prices"] # (N, T, 5), 'close', 'high', 'low', 'open', 'volume'
            timestamps = history["timestamps"] # (N, T)
            prices_mean = history["prices_mean"] # (N, T, 5), 'close', 'high', 'low', 'open', 'volume'
            prices_std = history["prices_std"] # (N, T, 5), 'close', 'high', 'low', 'open', 'volume'

            # Concat features
            features = torch.cat([prices, features, times], dim=-1)  # (N, T, 5 + F + 4)

            # Remove the last volume
            volumes = original_prices[:, :, -1]  # (N, T)
            original_prices = original_prices[:, :, :-1]  # (N, T, 4)
            prices = prices[:, :, :-1]  # (N, T, 4)
            prices_mean = prices_mean[:, :, :-1]  # (N, T, 4)
            prices_std = prices_std[:, :, :-1]  # (N, T, 4)

            features = features.to(self.device, self.dtype)
            prices = prices.to(self.device, self.dtype)
            prices_mean = prices_mean.to(self.device, self.dtype)
            prices_std = prices_std.to(self.device, self.dtype)

            if len(features.shape) == 3:
                features = features.unsqueeze(1) # (N, C, T, F)
            if len(prices.shape) == 3:
                prices = prices.unsqueeze(1) # (N, C, T, 4)
            if len(original_prices.shape) == 3:
                original_prices = original_prices.unsqueeze(1) # (N, C, T, 4)
            if len(prices_mean.shape) == 3:
                prices_mean = prices_mean.unsqueeze(1) # (N, C, T, 4)
            if len(prices_std.shape) == 3:
                prices_std = prices_std.unsqueeze(1) # (N, C, T, 4)
            if len(labels.shape) == 3:
                labels = labels.unsqueeze(1) # (N, C, T, 2)
            # The next day returns of the last day
            labels = labels[:, :, -1, 0] # (N, C)

            # Forward
            if if_train:
                output = self.model(features)
            else:
                with torch.no_grad():
                    output = self.model(features)

            pred_prices = output["recon_sample"] # (N, C, T, 4), 'close', 'high', 'low', 'open'
            posterior = output["posterior"]

            # Restore prices
            restored_target_prices = original_prices
            restored_pred_prices = pred_prices * prices_std + prices_mean # (N, C, T, 4)

            input_size = prices.shape[2:]
            patch_size = tuple(list(self.patch_size[:-1]) + [input_size[-1]])
            patch_info = get_patch_info(self.input_channel, input_size, patch_size)

            patched_target_prices = patchify(prices, patch_info=patch_info)  # (N, L, D)
            patched_pred_prices = patchify(pred_prices, patch_info=patch_info)  # (N, L, D)

            # plot
            if if_plot and data_iter_step in sample_batchs:
                if self.is_main_process:
                    # try:
                        save_prefix = "{}_epoch_{:06d}_batch_{:06d}.jpg".format(mode, epoch, data_iter_step)
                        save_path = os.path.join(self.plot_path, save_prefix)
                        self._sample_plot(
                            start_timestamp,
                            end_timestamp,
                            timestamps.detach().cpu().numpy(),
                            restored_target_prices.squeeze(1).detach().cpu().numpy(),
                            restored_pred_prices.squeeze(1).detach().cpu().numpy(),
                            volumes=volumes.detach().cpu().numpy(),
                            save_path = save_path,
                        )
                    # except Exception as e:
                    #     logger.error(f"Plot error: {e}")

            mask = output["mask"]

            if self.vae_loss_fn:
                loss_dict = self.vae_loss_fn(sample=patched_pred_prices,
                                             target_sample=patched_target_prices,
                                             posterior=posterior,
                                             mask=mask,
                                             if_mask=if_mask)

                weighted_nll_loss = loss_dict["weighted_nll_loss"]
                weighted_kl_loss = loss_dict["weighted_kl_loss"]

                loss += weighted_nll_loss + weighted_kl_loss

                records.update({
                    "weighted_nll_loss": self.accelerator.gather(weighted_nll_loss).mean().item(),
                    "weighted_kl_loss": self.accelerator.gather(weighted_kl_loss).mean().item()
                })

            if self.price_cont_loss_fn:
                loss_dict = self.price_cont_loss_fn(prices=restored_pred_prices.squeeze(1))
                weighted_cont_loss = loss_dict["weighted_cont_loss"]

                records.update({
                    "weighted_cont_loss": self.accelerator.gather(weighted_cont_loss).mean().item()
                })

                loss += weighted_cont_loss

            records.update({
                "loss": self.accelerator.gather(loss).mean().item()
            })

            restored_pred_prices = patchify(restored_pred_prices, patch_info = patch_info)
            restored_target_prices = patchify(restored_target_prices, patch_info = patch_info)
            restored_pred_prices = restored_pred_prices.detach().cpu().numpy()
            restored_target_prices = restored_target_prices.detach().cpu().numpy()

            if if_mask:

                mask = mask.detach().cpu().numpy()
                mask = mask.repeat(1, prices.shape[-1])
                mask_target_prices = restored_target_prices * mask
                mask_pred_prices = restored_pred_prices * mask
                nomask_target_prices = restored_target_prices * (1.0 - mask)
                nomask_pred_prices = restored_pred_prices * (1.0 - mask)

                mask_mse = MSE(mask_target_prices, mask_pred_prices)
                nomask_mse = MSE(nomask_target_prices, nomask_pred_prices)
                mse = MSE(restored_target_prices, restored_pred_prices)

                records.update({
                    "mask_mse": float(mask_mse),
                    "nomask_mse": float(nomask_mse),
                    "mse": float(mse)
                })

            else:
                mse = MSE(restored_target_prices, restored_pred_prices)

                records.update({
                    "mse": float(mse)
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
                        self.writer.add_scalar(f"{prefix}/{key}", value, global_step)
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
        logger.info(log_str)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self):

        logger.info("| Start training and evaluating VAE...")

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
        logger.info("| Start testing VAE...")

        logger.info("| Load the last checkpoint.")
        epoch = self.load_checkpoint()

        log_stats = {"epoch": epoch}

        train_stats = self.run_step(epoch,
                                   mode="train",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=True)

        valid_stats = self.run_step(epoch,
                                   mode="valid",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=True)

        test_stats = self.run_step(epoch,
                                   mode="test",
                                   if_use_writer=False,
                                   if_use_wandb=False,
                                   if_plot=True)

        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
        log_stats.update({f"valid_{k}": v for k, v in valid_stats.items()})
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "test_log.txt"), "w",) as f:
                f.write(json.dumps(log_stats) + "\n")

        logger.info("| Test finished.")