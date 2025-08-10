#!/usr/bin/env bash

# Kill any existing SAC trading processes
# ps -ef | grep sac | grep -v grep | awk '{print $2}' | xargs kill -9

CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/AAPL_sac_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/AMZN_sac_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/GOOGL_sac_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/META_sac_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/MSFT_sac_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/sac/TSLA_sac_trading.py &