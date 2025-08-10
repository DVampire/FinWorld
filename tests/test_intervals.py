import argparse
import os
import sys
from pathlib import Path
from mmengine import DictAction
from datetime import datetime

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.log import logger
from finworld.config import config
from finworld.utils import generate_intervals

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "download", "day_level_price", "exp.py"), help="config file path")

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

if __name__ == "__main__":

    # Parse command line arguments
    args = parse_args()

    # Initialize the configuration
    config.init_config(args.config, args)

    # Initialize the logger
    logger.init_logger(log_path=config.log_path, accelerator=None)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config}")

    start_date = datetime.strptime(config.downloader.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(config.downloader.end_date, "%Y-%m-%d")

    # Generate intervals
    intervals = generate_intervals(start_date, end_date, interval_level='day')
    logger.info(f"| {intervals}")

