import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(verbose=True)
import pandas as pd

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.log import logger
from finworld.factor import Alpha158

def main():
    df = pd.read_json(os.path.join(root, "workdir/exp_fmp_price_1day/price/AAPL.jsonl"), lines=True)

    factor = Alpha158()
    df, factors_info = factor.run(df, windows=[5, 10, 20, 30, 60], level='1day')

if __name__ == "__main__":

    # Initialize the logger
    log_path = os.path.join(root, "workdir", "tests")
    os.makedirs(log_path, exist_ok=True)

    log_path = os.path.join(log_path, "logger.log")
    logger.init_logger(log_path=log_path, accelerator=None)

    main()