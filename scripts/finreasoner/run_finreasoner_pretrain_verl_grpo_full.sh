#!/bin/bash
set -x

# Git
git pull -f

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export TP_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=XFORMERS

TAG=${TAG:-"verl_qwen2.5-7b_grpo_full"}
PROJECT_NAME="finreasoner"
EXPERIMENT_NAME=${TAG}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKDIR="${ROOT}/workdir/${PROJECT_NAME}/${EXPERIMENT_NAME}"
MODEL_PATH="${ROOT}/hub/Qwen2.5-7B"
CHECKPOINT_PATH="${WORKDIR}/checkpoint"
WANDB_API_KEY=4025943f5c98398d235eae04243f882b45bcd591
WANDB_ENTITY=zwt963
WANDB_BASE_URL=https://api.bandw.top

mkdir -p "${CHECKPOINT_PATH}"

TRAIN_FILES="['${ROOT}/datasets/finreasoner/flare-finqa_train.parquet', '${ROOT}/datasets/finreasoner/salesforce_train.parquet', '${ROOT}/datasets/finreasoner/fineval_train.parquet', '${ROOT}/datasets/finreasoner/convfinqa_train.parquet', '${ROOT}/datasets/finreasoner/finance_exam_train.parquet', '${ROOT}/datasets/finreasoner/cflue_train.parquet']"
TEST_FILES="['${ROOT}/datasets/finreasoner/flare-finqa_test.parquet','${ROOT}/datasets/finreasoner/convfinqa_test.parquet','${ROOT}/datasets/finreasoner/fineval_test.parquet','${ROOT}/datasets/finreasoner/cflue_test.parquet']"

# Parameters:
train_batch_size=64
max_prompt_length=4096
max_response_length=4096
max_num_batched_tokens=$((max_prompt_length + max_response_length))
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
truncation=left
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=16
tensor_model_parallel_size=4
rolloutn=8
n_gpus_per_node=8
nnodes=2
save_freq=100
test_freq=10
total_epochs=50
offload=True

python3 -m finworld.mverl.simple.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TEST_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=${truncation} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rolloutn} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    wandb.api_key=${WANDB_API_KEY} \
    wandb.entity=${WANDB_ENTITY} \
    wandb.base_url=${WANDB_BASE_URL} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.default_local_dir="${CHECKPOINT_PATH}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs}