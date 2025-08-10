import os
import numpy as np
import json
from copy import deepcopy

from finworld.registry import TRAINER
from finworld.registry import ENVIRONMENT
from finworld.trainer.base import Trainer
from finworld.log import logger
from finworld.utils import TradingRecords

@TRAINER.register_module(force=True)
class RuleTrdingTrainer(Trainer):
    def __init__(self,
                 config = None,
                 dataset=None,
                 metrics = None,
                 agent=None,
                 **kwargs
                 ):
        super(RuleTrdingTrainer, self).__init__(**kwargs)

        self.config = config

        # Build the environments
        train_environment_config = deepcopy(config.train_environment)
        train_environment_config.update({"dataset": dataset})
        self.train_environment = ENVIRONMENT.build(train_environment_config)

        valid_environment_config = deepcopy(config.valid_environment)
        valid_environment_config.update({"dataset": dataset})
        self.valid_environment = ENVIRONMENT.build(valid_environment_config)

        test_environment_config = deepcopy(config.test_environment)
        test_environment_config.update({"dataset": dataset})
        self.test_environment = ENVIRONMENT.build(test_environment_config)

        self.agent = agent
        self.metrics = metrics

        self.exp_path = self.config.exp_path

        self._init_params()

    def _init_params(self):
        logger.info(f"| Initializing parameters...")

        # Set random seed for reproducibility
        # DO NOTHING

        logger.info(f"| Parameters initialized successfully.")

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}()"

    def __repr__(self):
        return self.__str__()

    def train(self):
        logger.warning("RuleTradingTrainer does not support train method. Please use valid instead.")

    def inference(self, environment = None, mode: str = "valid"):

        trading_records = TradingRecords()

        state, info = environment.reset()

        data = environment.get_data()
        original_prices_df = data["original_prices_df"]
        actions = self.agent.forward(original_prices_df)

        timestamp_string = info["timestamp"]

        trading_records.add(
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
            state, reward, done, truncted, info = environment.step(action)

            timestamp_string = info["timestamp"]

            trading_records.add(
                dict(
                    action=info["action"],
                    action_label=info["action_label"],
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
        trading_records.add(
            dict(
                action=info["action"],
                action_label=info["action_label"],
                ret=info["ret"],
                total_profit=info["total_profit"],
            )
        )

        rets = trading_records.data["ret"]
        rets = np.array(rets, dtype=np.float32)

        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            arguments = dict(
                ret=rets,
            )
            metric_item = metric_fn(**arguments)
            metrics[metric_name] = metric_item

        # Save the trading records
        records_df = trading_records.to_dataframe()
        records_df.to_csv(os.path.join(self.exp_path, f"{mode}_records.csv"), index=True)

        log_string = f"{mode.capitalize()} Metrics:\n"
        metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"| {log_string} - {metrics_string}")

        return metrics

    def valid(self):

        stats = {}

        logger.info(f"| Validating RuleAgent...")
        valid_stats = self.inference(environment=self.valid_environment, mode="valid")
        stats.update(valid_stats)

        with open(os.path.join(self.exp_path, "valid_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Validation completed.")

    def test(self):

        stats = {}

        logger.info(f"| Testing RuleAgent...")
        test_stats = self.inference(environment=self.test_environment, mode="test")
        stats.update(test_stats)

        with open(os.path.join(self.exp_path, "test_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Testing completed.")