#!/bin/bash
# CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=29510 --nproc_per_node=2 scripts/time/train.py --config configs/time/exp_autoformer.py
# CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=29511 --nproc_per_node=2 scripts/time/train.py --config configs/time/dj30_autoformer.py
# CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=29512 --nproc_per_node=2 scripts/time/train.py --config configs/time/dj30_crossformer.py
# CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=29513 --nproc_per_node=2 scripts/time/train.py --config configs/time/dj30_dlinear.py
CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=29514 --nproc_per_node=2 scripts/time/train.py --config configs/time/dj30_etsformer.py