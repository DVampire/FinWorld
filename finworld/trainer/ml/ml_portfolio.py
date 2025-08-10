import os
import numpy as np
import json
from copy import deepcopy

from finworld.registry import TRAINER
from finworld.registry import ENVIRONMENT
from finworld.registry import DOWNSTREAM
from finworld.trainer.base import Trainer
from finworld.log import logger
from finworld.utils import PortfolioRecords

@TRAINER.register_module(force=True)
class MLPortfolioTrainer(Trainer):
    def __init__(self,
                 config = None,
                 dataset=None,
                 metrics = None,
                 agent=None,
                 **kwargs
                 ):
        super(MLPortfolioTrainer, self).__init__(**kwargs)

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
        self.agent = agent
        self.metrics = metrics

        self.exp_path = self.config.exp_path

        self.data_columns = dataset.assets_meta_info['columns']

        self._init_params()

    def _init_params(self):
        logger.info(f"| Initializing parameters...")

        # DO NOTHING

        logger.info(f"| Parameters initialized successfully.")

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}()"

    def __repr__(self):
        return self.__str__()

    def train(self):

        logger.info(f"| Training MLPortfolioAgent...")

        train_data = self.train_environment.get_data()
        valid_data = self.valid_environment.get_data()

        features_dfs = train_data["features_dfs"]
        labels_dfs = train_data["labels_dfs"]

        labels_columns = self.data_columns["labels"]
        ret_labels = [col for col in labels_columns if col.startswith("ret")]

        data = {}
        data["features"] = features_dfs
        data["labels"] = {key: value[ret_labels] for key, value in labels_dfs.items()}

        valid_features_df = valid_data["features_dfs"]
        valid_labels_df = valid_data["labels_dfs"]
        valid_data = {}
        valid_data["features"] = valid_features_df
        valid_data["labels"] = {key: value[ret_labels] for key, value in valid_labels_df.items()}

        self.agent.train(data, valid_data)
        self.valid()

    def inference(self, environment = None, mode: str = "valid"):

        portfolio_records = PortfolioRecords()

        data = environment.get_data()
        features_dfs = data["features_dfs"]
        data = {
            "features": features_dfs,
        }

        pred = self.agent.forward(data)

        timestamp = features_dfs[list(features_dfs.keys())[0]].index
        timestamp = np.array([ts.strftime(self.downstream_environment.level_format.value) for ts in timestamp])

        actions = self.downstream(y_pred=pred, timestamp=timestamp)

        state, info = environment.reset()

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
            state, reward, done, truncted, info = environment.step(action)

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

        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            arguments = dict(
                ret=rets,
            )
            metric_item = metric_fn(**arguments)
            metrics[metric_name] = metric_item

        # Save the trading records
        records_df = portfolio_records.to_dataframe()
        records_df.to_csv(os.path.join(self.exp_path, f"{mode}_records.csv"), index=True)

        log_string = f"{mode.capitalize()} Metrics:\n"
        metrics_string = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"| {log_string} - {metrics_string}")

        return metrics

    def valid(self):

        stats = {}

        logger.info(f"| Validating MLPortfolioAgent...")
        valid_stats = self.inference(environment=self.valid_environment, mode="valid")
        stats.update(valid_stats)

        with open(os.path.join(self.exp_path, "valid_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Validation completed.")

    def test(self):

        stats = {}

        logger.info(f"| Testing MLPortfolioAgent...")
        test_stats = self.inference(environment=self.test_environment, mode="test")
        stats.update(test_stats)

        with open(os.path.join(self.exp_path, "test_stats.txt"), "a",) as f:
            f.write(json.dumps(stats) + "\n")

        logger.info(f"| Testing completed.")