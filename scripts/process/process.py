import argparse
import os
import sys
from pathlib import Path
from mmengine import DictAction
import asyncio

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from finworld.log import logger
from finworld.config import config
from finworld.registry import PROCESSOR

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "process", "dj30.py"), help="config file path")

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

async def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize the configuration
    config.init_config(args.config, args)

    # Initialize the logger3
    logger.init_logger(config=config)
    logger.info(f"| Logger initialized at: {config.log_path}")
    logger.info(f"| Config:\n{config.pretty_text}")

    processor = PROCESSOR.build(config.processor)

    try:
        await processor.run()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == '__main__':
    asyncio.run(main())