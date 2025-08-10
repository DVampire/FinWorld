#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python scripts/rl_portfolio/train.py --config=configs/rl_portfolio/sac/dj30_sac_portfolio.py &