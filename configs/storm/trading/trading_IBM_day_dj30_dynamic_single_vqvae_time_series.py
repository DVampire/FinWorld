select_asset = "IBM"
workdir = "workdir"
tag = f"trading_{select_asset}_day_dj30_dynamic_single_vqvae_time_series"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"
project = "storm"
tensorboard_path = "tensorboard"
checkpoint_path = "checkpoint"
wandb_path = "wandb"
history_timestamps = 64
future_timestamps = 32
timestamp_format = "%Y-%m-%d"

# training parameters (adjust mainly)
policy_learning_rate = 5e-5
value_learning_rate = 1e-5
num_steps = 128
policy_num_minibatches = 128
value_num_minibatches = 16
gradient_checkpointing_steps = 32
total_timesteps = int(1e8)
check_steps = int(1e4)
seed = 2024

# fixed parameters of training [do not change]
state_shape = (16, 512)
dtype = "fp32"
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
num_envs = 4
embed_dim = 128
action_dim = 3
depth = 2
gae_lambda = 0.95
batch_size = int(num_envs * num_steps)
policy_minibatch_size = int(batch_size // policy_num_minibatches)
value_minibatch_size = int(batch_size // value_num_minibatches)
anneal_lr = True
clip_vloss = True
norm_adv = True
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
max_grad_norm = 0.5
critic_warm_up_steps = 0
target_kl = 0.02
gamma = 0.99
update_epochs = 1

transition = ["states", "actions", "logprobs", "rewards", "dones", "values"]
transition_shape = dict(
    states = dict(shape = (num_envs, state_shape[0], state_shape[1]), type = "float32"),
    actions = dict(shape = (num_envs,), type = "int32"),
    logprobs =dict(shape=(num_envs,), type="float32"),
    rewards = dict(shape = (num_envs, ), type = "float32"),
    dones = dict(shape = (num_envs, ), type = "float32"),
    values =dict(shape=(num_envs,), type="float32"),
)

dataset = dict(
    type = "StateDataset",
    data_path="datasets/processd_day_dj30/features",
    assets_path="configs/_asset_list_/dj30.json",
    fields_name={
        "prices": [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
        ],
    },
    states_path="workdir/pretrain_day_dj30_dynamic_single_vqvae_time_series/state.joblib",
    select_asset=select_asset,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    start_timestamp="2008-04-01",
    end_timestamp="2024-04-01",
    timestamp_format = "%Y-%m-%d",
    if_use_cs = False,
    if_use_ts = True,
    exp_path = exp_path,
)

environment = dict(
    type="Environment",
    mode="train",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    timestamp_format=timestamp_format,
    start_timestamp="2008-04-01",
    end_timestamp="2024-04-01",
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
)

train_environment = environment.copy()
train_environment.update(
    mode="train",
    dataset=None,
    start_timestamp="2008-04-01",
    end_timestamp="2021-04-01",
)

valid_environment = environment.copy()
valid_environment.update(
    mode="valid",
    dataset=None,
    start_timestamp="2021-04-01",
    end_timestamp="2024-04-01",
)

test_environment = environment.copy()
test_environment.update(
    mode="test",
    dataset=None,
    start_timestamp="2021-04-01",
    end_timestamp="2024-04-01",
)

agent = dict(
    type="PPO",
    input_size = state_shape,
    embed_dim = embed_dim,
    depth = depth,
    action_dim = action_dim,
)