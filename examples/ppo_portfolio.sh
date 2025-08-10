#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/ppo/dj30_ppo_portfolio.py &
CUDA_VISIBLE_DEVICES=5 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/ppo/sse50_ppo_portfolio.py &
CUDA_VISIBLE_DEVICES=5 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/ppo/hs300_ppo_portfolio.py &
CUDA_VISIBLE_DEVICES=5 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/ppo/sp500_ppo_portfolio.py &