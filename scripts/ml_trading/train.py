import argparse
import os
import sys
from pathlib import Path
from mmengine import DictAction
from copy import deepcopy

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from finworld.log import logger
from finworld.config import config
from finworld.models import model_manager
from finworld.registry import AGENT
from finworld.registry import DATASET
from finworld.registry import TRAINER
from finworld.registry import TASK
from finworld.registry import METRIC

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "ml_trading", "lightgbm", "AAPL_lightgbm_trading.py"), help="config file path")
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

    # Initialize the logger
    logger.init_logger(config=config)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")

    # Registed models
    model_manager.init_models(use_local_proxy=False)
    logger.info("Registed models: %s", ", ".join(model_manager.registed_models.keys()))

    # Build the dataset
    dataset = DATASET.build(config.dataset)

    # Build metrics
    metric_configs = config.metric
    metrics = {}
    for metric_name, metric_config in metric_configs.items():
        metric_config.update(
            {
                "level": config.level,
                "symbol_info": dataset.asset_info,
            }
        )
        metrics[metric_name] = METRIC.build(metric_config)
    logger.info(f"| Metrics: \n{metrics}")

    # Create agent
    agent_config = config.agent
    agent = AGENT.build(agent_config)
    logger.info(f"| Agent: \n{agent}")

    # Build trainer
    trainer_config = config.trainer
    trainer_config.update(dict(
        config=config,
        dataset=dataset,
        agent=agent,
        metrics=metrics,
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

    task.run()

if __name__ == '__main__':
    main()