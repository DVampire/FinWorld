#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/AAPL_ppo_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/AMZN_ppo_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/GOOGL_ppo_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/META_ppo_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/MSFT_ppo_trading.py &
CUDA_VISIBLE_DEVICES=4 python scripts/rl_trading/train.py --config=configs/rl_trading/ppo/TSLA_ppo_trading.py &