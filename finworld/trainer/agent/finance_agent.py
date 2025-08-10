import os
import numpy as np
import json

from finworld.registry import TRAINER
from finworld.trainer.base import Trainer
from finworld.log import logger
from finworld.utils import TradingRecords

@TRAINER.register_module(force=True)
class FinanceAgentTrainer(Trainer):
    def __init__(self,
                 config = None,
                 train_environment=None,
                 valid_environment=None,
                 test_environment=None,
                 metrics = None,
                 agent=None,
                 **kwargs
                 ):
        super(FinanceAgentTrainer, self).__init__(**kwargs)

        self.config = config
        self.train_environment = train_environment
        self.valid_environment = valid_environment
        self.test_environment = test_environment
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
        logger.warning("FinAgentTrainer does not support train method. Please use valid instead.")

    async def inference(self, environment = None, mode: str = "valid"):

        trading_records = TradingRecords()

        state, info = environment.reset()
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

            res = await self.agent.run(state=state, info=info, reset=False)
            action = res.output
            state, reward, done, truncted, info = environment.step(action)

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

            if "final_info" in info:
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

    async def valid(self):

        stats = {}

        logger.info(f"| Validating FinAgent...")
        valid_stats = await self.inference(environment=self.valid_environment, mode="valid")
        stats.update(valid_stats)

        with open(os.path.join(self.exp_path, "valid_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Validation completed.")

    async def test(self):

        stats = {}

        logger.info(f"| Testing FinAgent...")
        test_stats = await self.inference(environment=self.test_environment, mode="test")
        stats.update(test_stats)

        with open(os.path.join(self.exp_path, "test_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Testing completed.")