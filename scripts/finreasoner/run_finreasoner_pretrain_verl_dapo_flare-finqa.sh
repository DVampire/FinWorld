#!/bin/bash
set -x

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export TP_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=XFORMERS

TAG=${TAG:-"verl_qwen3-8b_dapo_flare-finqa"}
PROJECT_NAME="finreasoner"
EXPERIMENT_NAME=${TAG}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR="${ROOT}/workdir/${PROJECT_NAME}/${EXPERIMENT_NAME}"
MODEL_PATH="${ROOT}/hub/Qwen3-8B"
CHECKPOINT_PATH="${WORKDIR}/checkpoint"

mkdir -p "${CHECKPOINT_PATH}"

TRAIN_FILES="['${ROOT}/datasets/finreasoner/flare-finqa_train.parquet']"
TEST_FILES="['${ROOT}/datasets/finreasoner/flare-finqa_test.parquet']"

# Parameters
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
max_prompt_length=4096
max_response_length=4096
enable_overlong_buffer=True
overlong_buffer_len=8192
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_batch_size=64
gen_batch_size=$((train_batch_size * 3))
rolloutn=8
ppo_mini_batch_size=32
# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
# Performance Related Parameter
ulysses_sequence_parallel_size=1
use_dynamic_bsz=True
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
offload=True
tensor_model_parallel_size=2
rolloutn=8
save_freq=100
test_freq=10
total_epochs=30

python3 -m finworld.mverl.dapo.trainer.main_dapo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_batch_size} \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.rollout.n=${rolloutn} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CHECKPOINT_PATH}" \
    trainer.resume_mode=auto