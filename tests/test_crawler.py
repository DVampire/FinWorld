import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.log import logger
from finworld.utils import fetch_url

async def main():
    url = "https://seekingalpha.com/article/4247474-a-sweet-dividend-hike-coming-from-apple?source=feed"
    result = await fetch_url(
        url=url,
    )
    content = result.markdown if result else "Failed to fetch content"
    return content

if __name__ == "__main__":

    # Initialize the logger
    log_path = os.path.join(root, "workdir", "tests")
    os.makedirs(log_path, exist_ok=True)

    log_path = os.path.join(log_path, "logger.log")
    logger.init_logger(log_path=log_path, accelerator=None)

    result = asyncio.run(main())
    logger.info(result)