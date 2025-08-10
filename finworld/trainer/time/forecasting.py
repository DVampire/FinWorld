import pandas as pd
import torch
from typing import Dict
from tensordict import TensorDict
import os
import numpy as np
import json
from glob import glob
from copy import deepcopy

from finworld.registry import TRAINER
from finworld.registry import DATALOADER
from finworld.registry import DOWNSTREAM
from finworld.registry import ENVIRONMENT
from finworld.trainer.base import Trainer
from finworld.log import logger
from finworld.utils import check_data
from finworld.utils import MetricLogger, SmoothedValue
from finworld.utils import is_main_process
from finworld.utils import NativeScalerWithGradNormCount as NativeScaler
from finworld.utils import cpu_mem_usage, gpu_mem_usage
from finworld.utils import PortfolioRecords
from finworld.task import TaskType
from finworld.metric import TRADING_METRICS, REGRESSION_METRICS


@TRAINER.register_module(force=True)
class ForecastingTrainer(Trainer):
    def __init__(self,
                 config,
                 model,
                 collate_fn,
                 dataset,
                 optimizer,
                 scheduler,
                 losses,
                 metrics,
                 plot,
                 device,
                 dtype,
                 **kwargs):

        super(ForecastingTrainer, self).__init__(**kwargs)
        
        self.task_type = TaskType.from_string(config.task_type)

        self.config = config
        self.model = model
        self.collect_fn = collate_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = losses
        self.metrics = metrics

        # Build the dataset and dataloader
        train_dataset = deepcopy(dataset)
        train_dataset.crop(
            start_timestamp=config.train_dataset.start_timestamp,
            end_timestamp=config.train_dataset.end_timestamp,
        )
        logger.info(f"| Train dataset: \n{train_dataset}")
        train_dataloader_config = deepcopy(config.train_dataloader)
        train_dataloader_config.update({
            "collate_fn": collate_fn,
            "dataset": train_dataset,
        })
        self.train_dataloader = DATALOADER.build(train_dataloader_config)

        valid_dataset = deepcopy(dataset)
        valid_dataset.crop(
            start_timestamp=config.valid_dataset.start_timestamp,
            end_timestamp=config.valid_dataset.end_timestamp,
        )
        logger.info(f"| Valid dataset: \n{valid_dataset}")
        valid_dataloader_config = deepcopy(config.valid_dataloader)
        valid_dataloader_config.update({
            "collate_fn": collate_fn,
            "dataset": valid_dataset,
        })
        self.valid_dataloader = DATALOADER.build(valid_dataloader_config)

        test_dataset = deepcopy(dataset)
        test_dataset.crop(
            start_timestamp=config.test_dataset.start_timestamp,
            end_timestamp=config.test_dataset.end_timestamp,
        )
        logger.info(f"| Test dataset: \n{test_dataset}")
        test_dataloader_config = deepcopy(config.test_dataloader)
        test_dataloader_config.update({
            "collate_fn": collate_fn,
            "dataset": test_dataset,
        })
        self.test_dataloader = DATALOADER.build(test_dataloader_config)

        self.trading_metrics = {
           name: metric for name, metric in metrics.items() if isinstance(metric, tuple(TRADING_METRICS))
        }
        self.reg_metrics = {
            name: metric for name, metric in metrics.items() if isinstance(metric, tuple(REGRESSION_METRICS))
        }

        # Build downstream strategy
        if hasattr(config, "downstream"):
            downstream_config = config.downstream
            downstream = DOWNSTREAM.build(downstream_config)
            logger.info(f"| Downstream: \n{downstream}")

            downstream_environment_config = config.downstream_environment
            downstream_environment_config.update({"dataset": dataset})
            downstream_environment = ENVIRONMENT.build(downstream_environment_config)
            logger.info(f"| Downstream Environment: \n{downstream_environment}")
        else:
            downstream = None
            downstream_environment = None

        self.downstream_environment = downstream_environment
        self.downstream = downstream
        self.plot = plot
        self.device = device
        self.dtype = dtype

        self.plot_path = self.config.plot_path
        self.checkpoint_path = self.config.checkpoint_path
        self.check_batch_info_flag = True
        self.checkpoint_period = self.config.checkpoint_period
        self.num_checkpoint_del = self.config.num_checkpoint_del
        self.print_freq = self.config.print_freq
        self.exp_path = self.config.exp_path
        self.num_plot_samples = self.config.num_plot_samples
        self.num_plot_samples_per_batch = self.config.num_plot_samples_per_batch
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"
        self.num_training_epochs = self.config.num_training_epochs

        self._init_params()

        self.start_epoch = self.load_checkpoint() + 1

        self.data_columns = train_dataset.assets_meta_info['columns']

    def _init_params(self):
        logger.info(f"| Initializing parameters...")

        self.is_main_process = is_main_process()

        self.loss_scaler = NativeScaler(fp32=self.config.fp32)

        torch.set_default_dtype(self.dtype)

        logger.info(f"| Parameters initialized successfully.")

    def save_checkpoint(self, epoch: int):
        if not self.is_main_process:
            return  # Only save checkpoint on the main process

        checkpoint_file = os.path.join(self.checkpoint_path, "checkpoint_{:06d}.pth".format(epoch))

        # Save model, optimizer, and scheduler states
        state = {
            'epoch': epoch,
            'model_state': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        # Save the state to a file
        torch.save(state, checkpoint_file)

        # Manage saved checkpoints
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(checkpoint_files) > self.num_checkpoint_del:
            for checkpoint_file in checkpoint_files[:-self.num_checkpoint_del]:
                os.remove(checkpoint_file)
                logger.info(f"｜ Checkpoint deleted: {checkpoint_file}")

        logger.info(f"| Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, epoch: int = -1):

        epoch_checkpoint_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch:06d}.pth")

        latest_checkpoint_file = None
        checkpoint_files = glob(os.path.join(self.checkpoint_path, "checkpoint_*.pth"))
        if not checkpoint_files:
            logger.info(f"| No checkpoint found in {self.checkpoint_path}.")
        else:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_file = checkpoint_files[-1]
            latest_checkpoint_file = checkpoint_file

        if epoch >= 0 and os.path.exists(epoch_checkpoint_file):
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

            if self.config.distributed:
                self.model.module.load_state_dict(state['model_state'])
            else:
                self.model.load_state_dict(state['model_state'])

            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])

            return state['epoch']
        else:
            return 0

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(num_training_epochs={self.config.num_training_epochs}, batch_size={self.config.batch_size})"

    def __repr__(self):
        return self.__str__()

    def check_batch_info(self, batch: Dict):

        if self.check_batch_info_flag:
            assets = batch["assets"]
            logger.info(f"| Asset: {check_data(assets)}")

            for key in batch:
                if key not in ["assets"]:
                    data = batch[key]
                    log_str = f"| {key}: "
                    for key, value in data.items():
                        log_str += f"\n {key}: {check_data(value)}"
                    logger.info(log_str)

            self.check_batch_info_flag = False

    def _prepare_batch(self, batch: Dict):
        """
        Prepare the batch for training.
        """
        history = batch["history"]
        future = batch["future"]

        history_start_timestamp = history["start_timestamp"]  # (N,)
        history_end_timestamp = history["end_timestamp"]  # (N,)
        history_start_index = history["start_index"]  # (N,)
        history_end_index = history["end_index"]  # (N,)
        history_prices = history["prices"]  # (N, T, S, 5), 'close', 'high', 'low', 'open', 'volume'
        history_features = history["features"]  # (N, T, S, F)
        history_labels = history["labels"]  # (N, T, S, 2)
        history_times = history["times"]  # (N, T, S, 4), 'day', 'month', 'weekday', 'year'
        history_original_prices = history["original_prices"]  # (N, T, S, 5), 'close', 'high', 'low', 'open', 'volume'
        history_timestamps = history["timestamps"]  # (N, T)
        history_prices_mean = history["prices_mean"]  # (N, T, S, 5), 'close', 'high', 'low', 'open', 'volume'
        history_prices_std = history["prices_std"]  # (N, T, S, 5), 'close', 'high', 'low', 'open', 'volume'

        future_start_timestamp = future["start_timestamp"]  # (N,)
        future_end_timestamp = future["end_timestamp"]
        future_start_index = future["start_index"]
        future_end_index = future["end_index"]
        future_prices = future["prices"]
        future_features = future["features"]
        future_labels = future["labels"]
        future_times = future["times"]
        future_original_prices = future["original_prices"]
        future_timestamps = future["timestamps"]
        future_prices_mean = future["prices_mean"]
        future_prices_std = future["prices_std"]

        future_times = torch.concat([
            history_times[:, -self.config.review_timestamps:, ...], future_times,
        ], dim=1)

        history_features = history_features.to(torch.float32)
        future_features = future_features.to(torch.float32)
        history_times = history_times.to(torch.long)
        future_times = future_times.to(torch.long)

        x = TensorDict(
            {
                "features": history_features,
                "times": history_times,
            }, batch_size=history_features.shape[0]).to(self.device)
        target = TensorDict(
            {
                "features": future_features,
                "times": future_times,
            }, batch_size=future_features.shape[0]).to(self.device)

        batch = dict(
            x = x,
            target = target,
        )

        labels_columns = self.data_columns["labels"]
        ret_labels_indices = [labels_columns.index(col) for col in labels_columns if col.startswith("ret")]
        future_labels = future_labels[:, 0, :, ret_labels_indices]
        future_labels = future_labels.permute(0, 2, 1) # (N, T, S) # N, [ret0001, ret0002, ...], S

        extra_batch = TensorDict(dict(
            future_start_timestamp = future_start_timestamp,
            history_end_timestamp = history_end_timestamp,
            future_prices = future_prices[..., :-1], # Remove the volume dimension
            future_labels = future_labels
        ), batch_size=future_prices.shape[0]).to(self.device)

        return batch, extra_batch

    def train_epoch(self,
                    epoch: int,
                    num_epochs: int = None,
                    dataloader = None,
                    mode: str = "train"
                    ):

        self.model.train()

        records = dict()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = f"Train Epoch: [{epoch}/{num_epochs}]"

        for step, batch in enumerate(metric_logger.log_every(dataloader,
                                                             self.print_freq,
                                                             header,
                                                             logger)):

            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            self.check_batch_info(batch)

            # Prepare batch
            batch, extra_batch = self._prepare_batch(batch)
            y_true = extra_batch["future_labels"]

            # Forward pass
            with torch.cuda.amp.autocast(enabled=not self.config.fp32):
                outputs = self.model(**batch)
                y_pred = outputs

            # Compute losses
            for name, loss_fn in self.losses.items():
                arguments = dict(
                    y_pred=y_pred,
                    y_true=y_true
                )

                loss_item = loss_fn(**arguments)

                records[name] = loss_item.item()
                loss += loss_item

            loss_value = loss.item()
            records["loss"] = loss_value

            self.optimizer.zero_grad()
            self.loss_scaler(
                loss,
                self.optimizer,
                parameters=self.model.parameters(),
                update_grad=True,
                clip_grad=self.config.clip_grad,
            )
            if self.scheduler:
                self.scheduler.step()

            torch.cuda.synchronize()

            # Log metrics
            for name, metric_fn in self.reg_metrics.items():
                arguments = dict(
                    y_true=y_true,
                    y_pred=y_pred
                )
                metric_item = metric_fn(**arguments)

                records[name] = metric_item.item()

            lr = self.optimizer.param_groups[0]["lr"]
            records["lr"] = lr

            cpu_mem = cpu_mem_usage()[0]
            records["cpu_mem"] = cpu_mem
            cpu_mem_all = cpu_mem_usage()[1]
            records["cpu_mem_all"] = cpu_mem_all
            gpu_mem = gpu_mem_usage()
            records["gpu_mem"] = gpu_mem

            if self.is_main_process:
                if ((step + 1) % self.print_freq == 0) or (step + 1 == len(dataloader)):
                    batch_metrics = {
                        f"batch_{mode}/{key}": value for key, value in records.items()
                    }
                    logger.log_metric(batch_metrics)

            metric_logger.update(**records)

        metric_logger.synchronize_between_processes()

        epoch_metrics = {f"epoch_{mode}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
        if self.is_main_process:
            logger.log_metric(epoch_metrics)

            log_string = f"Epoch [{epoch}/{num_epochs}]"
            metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
            logger.info(f"| {log_string} - {metrics_string}")

        return epoch_metrics

    def inference_epoch(self,
                        epoch: int,
                        num_epochs: int = None,
                        dataloader = None,
                        mode: str = "valid"
                        ):

        self.model.eval()

        records = dict()

        metric_logger = MetricLogger(delimiter="  ")

        header = f"Inference Epoch: [{epoch}/{num_epochs}]"

        collected_pred = []
        collected_true = []
        collected_timestamp = []
        for step, batch in enumerate(metric_logger.log_every(dataloader,
                                                             self.print_freq,
                                                             header,
                                                             logger)):
            with torch.no_grad():
                loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

                self.check_batch_info(batch)

                # Prepare batch
                batch, extra_batch = self._prepare_batch(batch)
                y_true = extra_batch["future_labels"]
                end_timestamp = extra_batch["history_end_timestamp"]

                # Forward pass
                outputs = self.model(**batch)
                y_pred = outputs

            batch_pred = y_pred[:, 0, ...]  # (N, S, T)
            batch_true = y_true[:, 0, ...]  # (N, S, T)
            batch_timestamp = end_timestamp
            collected_pred.append(batch_pred.detach().cpu().numpy())
            collected_true.append(batch_true.detach().cpu().numpy())
            collected_timestamp.extend(batch_timestamp) # Append list of timestamps

            # Compute losses
            for name, loss_fn in self.losses.items():
                arguments = dict(
                    y_pred=y_pred,
                    y_true=y_true,
                )

                loss_item = loss_fn(**arguments)

                records[name] = loss_item.item()

                loss += loss_item

            loss_value = loss.item()
            records["loss"] = loss_value

            # Log metrics
            for name, metric_fn in self.reg_metrics.items():
                arguments = dict(
                    y_true=y_true,
                    y_pred=y_pred
                )
                metric_item = metric_fn(**arguments)

                records[name] = metric_item.item()

            if self.is_main_process:

                # Log metrics
                if ((step + 1) % self.print_freq == 0) or (step + 1 == len(dataloader)):
                    batch_metrics = {
                        f"batch_{mode}/{key}": value for key, value in records.items()
                    }

                    logger.log_metric(batch_metrics)

            metric_logger.update(**records)

        metric_logger.synchronize_between_processes()
        epoch_metrics = {f"epoch_{mode}/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}

        if self.is_main_process:

            if self.downstream:
                collected_pred = np.concatenate(collected_pred, axis=0)  # (N, S)
                collected_timestamp = np.array(collected_timestamp)  # (N,)
                sorted_idx = np.argsort(collected_timestamp)
                collected_pred = collected_pred[sorted_idx]  # Sort by timestamp
                collected_timestamp = collected_timestamp[sorted_idx]  # Sort by timestamp

                collected_timestamp = np.array([ts.strftime(self.downstream_environment.level_format.value)
                                       for ts in collected_timestamp])

                actions = self.downstream(y_pred=collected_pred, timestamp=collected_timestamp)

                portfolio_records = PortfolioRecords()
                state, info = self.downstream_environment.reset()

                timestamp_string = info["timestamp"]

                portfolio_records.add(
                    dict(
                        timestamp=info["timestamp"],
                        price=info["price"],
                        cash=info["cash"],
                        position=info["position"],
                        value=info["value"],
                    ),
                )

                while True:

                    action = actions[timestamp_string]
                    state, reward, done, truncted, info = self.downstream_environment.step(action)

                    timestamp_string = info["timestamp"]

                    portfolio_records.add(
                        dict(
                            action=info["action"],
                            ret=info["ret"],
                            total_profit=info["total_profit"],
                            timestamp=info["timestamp"],  # next timestamp
                            price=info["price"],  # next price
                            cash=info["cash"],  # next cash
                            position=info["position"],  # next position
                            value=info["value"],  # next value
                        ),
                    )

                    if done or truncted:
                        logger.info(f"| {mode.capitalize()} environment finished.")
                        break

                # End of the environment, add the final record
                portfolio_records.add(
                    dict(
                        action=info["action"],
                        ret=info["ret"],
                        total_profit=info["total_profit"],
                    )
                )

                rets = portfolio_records.data["ret"]
                rets = np.array(rets, dtype=np.float32)

                downstream_metrics = {}
                for metric_name, metric_fn in self.trading_metrics.items():
                    arguments = dict(
                        ret=rets,
                    )
                    metric_item = metric_fn(**arguments)
                    downstream_metrics[metric_name] = metric_item

                epoch_metrics.update({
                    f"epoch_{mode}/{k}": v for k, v in downstream_metrics.items()
                })

            logger.log_metric(epoch_metrics)

            log_string = f"Epoch [{epoch}/{num_epochs}]"
            metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
            logger.info(f"| {log_string} - {metrics_string}")

        return epoch_metrics

    def train(self):
        logger.info("| Start training...")

        for epoch in range(self.start_epoch, self.num_training_epochs + 1):

            if self.config.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            stats = {
                "epoch": epoch,
            }

            # Train for one epoch
            train_stats = self.train_epoch(epoch,
                                           num_epochs=self.num_training_epochs,
                                           dataloader=self.train_dataloader,
                                           mode="train")
            stats.update(train_stats)

            # Validate after each epoch
            valid_stats = self.inference_epoch(epoch,
                                               num_epochs=self.num_training_epochs,
                                               dataloader=self.valid_dataloader,
                                               mode="valid")
            stats.update(valid_stats)

            if self.is_main_process:
                with open(os.path.join(self.exp_path, "train_stats.txt"), "a",) as f:
                    f.write(json.dumps(stats) + "\n")

                if epoch % self.checkpoint_period == 0:
                    self.save_checkpoint(epoch)

        logger.info("| Training completed.")

    def valid(self):

        epoch = self.load_checkpoint()

        stats = {
            "epoch": epoch,
        }

        logger.info(f"| Validating at epoch {epoch}...")
        valid_stats = self.inference_epoch(epoch,
                                   num_epochs=1,
                                   dataloader=self.valid_dataloader,
                                   mode="valid")
        stats.update(valid_stats)

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "valid_stats.txt"), "a",) as f:
                f.write(json.dumps(stats) + "\n")
        logger.info(f"| Validation completed at epoch {epoch}.")

    def test(self):
        epoch = self.load_checkpoint()

        stats = {
            "epoch": epoch,
        }

        logger.info(f"| Testing at epoch {epoch}...")
        test_stats = self.inference_epoch(epoch,
                                  num_epochs=1,
                                  dataloader=self.test_dataloader,
                                  mode="test")
        stats.update(test_stats)

        if self.is_main_process:
            with open(os.path.join(self.exp_path, "test_stats.txt"), "a",) as f:
                f.write(json.dumps(stats) + "\n")
        logger.info(f"| Testing completed at epoch {epoch}.")