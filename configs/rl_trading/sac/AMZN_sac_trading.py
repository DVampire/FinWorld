task_type = "trading"
symbol = "AMZN"
workdir = "workdir"
method = "sac"
tag = f"{symbol}_{task_type}_{method}"
exp_path = f"{workdir}/{tag}"
project = "trading"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

#-----------------Dataset Parameters-----------------#
# dataset parameters
history_timestamps = 64
review_timestamps = 32
future_timestamps = 32
patch_timestamps = 4
start_timestamp = "1995-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2020-05-01"
if_norm = True
if_use_temporal = True
if_norm_temporal = False
if_use_future = False
level = "1day"

#----------------Environment Parameters------------#
max_count_sell = -1 # -1 means no limit
num_envs = 4
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
num_block_tokens = history_timestamps

#-----------------Model Parameters-----------------#
if_use_trajectory = True
if_use_sparse = True
dense_input_dim = 150
sparse_input_dim = 4 # days, months, weekdays, years
embedding_dim = 256
dropout = .0
num_heads = 4
depth = 4
action_dim = 3 # BUY, HOLD, SELL
actor_output_dim = 3 # BUY, HOLD, SELL
critic_output_dim = 1 # Value

#-----------------Trainer Parameters-----------------#
use_bc = False
use_expert= False
expert_weight = 0.3
kl_penalty_weight = 0.01
policy_learning_rate = 1e-5
value_learning_rate = 1e-6
num_steps = 128
policy_num_minibatches = 128
value_num_minibatches = 16
alpha_learning_rate = 1e-6
target_entropy_scale = 0.89
gradient_checkpointing_steps = 32
total_steps = int(1e8)
check_steps = int(1e4)
save_steps = int(1e5)
seed = 1024
batch_size = int(num_envs * num_steps)
policy_minibatch_size = int(batch_size // policy_num_minibatches)
value_minibatch_size = int(batch_size // value_num_minibatches)
warm_up_steps = 0
flash = True
gae_lambda = 0.95
anneal_lr = True
clip_vloss = True
norm_adv = True
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
bc_coef = 1e-3
max_grad_norm = 0.5
target_kl = 0.02
gamma = 0.99
update_epochs = 1
tau = 0.005
exploration_noise = 0.1
policy_noise = 0.2
replay_buffer_size = int(1e4)
learning_start = int(1e4)
device = "cuda"
dtype = "fp32"

#-----------------Transition Parameters-----------------#
# action-free data
indicator_transition = ["features", "times"] # indicator states
next_indicator_transition = ["next_features", "next_times"] # next indicator states

# action-dependent data
policy_trading_transition = [
    "policy_trading_cashes",
    "policy_trading_positions",
    "policy_trading_actions",
    "policy_trading_rets"
] # history policy states
next_policy_trading_transition = [
    "next_policy_trading_cashes",
    "next_policy_trading_positions",
    "next_policy_trading_actions",
    "next_policy_trading_rets",
] # next history policy states

expert_trading_transition = [
    "expert_trading_cashes",
    "expert_trading_positions",
    "expert_trading_actions",
    "expert_trading_rets",
]

# training data
training_transition = [
    "training_actions",
    "training_dones",
    "training_rewards",
    "training_expert_actions",
]

transition = (indicator_transition +
              next_indicator_transition +
              policy_trading_transition +
              next_policy_trading_transition +
              expert_trading_transition +
              training_transition)

transition_shape = dict(
    features=dict(shape=(num_envs, num_block_tokens, dense_input_dim), type="float32", low = -float("inf"), high = float("inf"), obs = True),
    times=dict(shape=(num_envs, num_block_tokens, sparse_input_dim), type="int32", low=0, high=float("inf"), obs=True),
    next_features=dict(shape=(num_envs, num_block_tokens, dense_input_dim), type="float32", low = -float("inf"), high = float("inf"), obs = False),
    next_times=dict(shape=(num_envs, num_block_tokens, sparse_input_dim), type="int32", low=0, high=float("inf"), obs=False),

    policy_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32",low = -float("inf"), high = float("inf"), obs = True),
    policy_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="float32", low = 0, high = float("inf"), obs = True),
    policy_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32", low = 0, high = action_dim - 1, obs = True),
    policy_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32", low = -10.0, high = 10.0, obs = True),

    next_policy_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32", low = -float("inf"), high = float("inf"), obs = False),
    next_policy_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="float32", low = 0, high = float("inf"), obs = False),
    next_policy_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32", low = 0, high = action_dim - 1, obs = False),
    next_policy_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32", low = -10.0, high = 10.0, obs = False),

    expert_trading_cashes=dict(shape=(num_envs, num_block_tokens,), type="float32", low=-float("inf"), high=float("inf"), obs=True),
    expert_trading_positions=dict(shape=(num_envs, num_block_tokens,), type="float32", low=0, high=float("inf"), obs=True),
    expert_trading_actions=dict(shape=(num_envs, num_block_tokens,), type="int32", low=0, high=action_dim - 1, obs=True),
    expert_trading_rets=dict(shape=(num_envs, num_block_tokens,), type="float32",  low=-10.0, high=10.0, obs=True),

    training_actions = dict(shape=(num_envs,), type="int32", obs = False),
    training_dones = dict(shape=(num_envs,), type="float32", obs = False),
    training_rewards = dict(shape = (num_envs, ), type = "float32", obs = False),
    training_expert_actions = dict(shape=(num_envs,), type="int32", obs = False),
)

# Tracker Configuration for accelerator
tracker = dict(
    tensorboard=dict(
        logging_dir = "logs/tensorboard",
    ),
    wandb=dict(
        project=project,
        name=tag,
        logging_dir="logs/wandb",
    ),
)

dataset = dict(
    type="SingleAssetDataset",
    symbol=symbol,
    data_path="datasets/exp",
    enabled_data_configs = [
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "price",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "feature",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "news",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "alpaca",
            "data_type": "news",
            "level": "1day",
        }
    ],
    if_norm=if_norm,
    if_use_future=if_use_future,
    if_use_temporal=if_use_temporal,
    if_norm_temporal=if_norm_temporal,
    scaler_cfg = dict(
        type="WindowedScaler"
    ),
    history_timestamps = history_timestamps,
    future_timestamps = future_timestamps,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    level=level,
)

environment = dict(
    type="EnvironmentPatchTrading",
    mode="train",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    history_timestamps=history_timestamps,
    patch_timestamps=patch_timestamps,
    future_timestamps=future_timestamps,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    max_count_sell=max_count_sell
)

train_environment = environment.copy()
train_environment.update(
    mode="train",
    dataset=None,
    start_timestamp=start_timestamp,
    end_timestamp=split_timestamp,
)

valid_environment = environment.copy()
valid_environment.update(
    mode="valid",
    dataset=None,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp,
)

test_environment = environment.copy()
test_environment.update(
    mode="test",
    dataset=None,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp,
)

metric = dict(
    arr = dict(
        type="ARR",
        level=level,
        symbol_info=None,
    ),
    sr=dict(
        type="SR",
        level=level,
        symbol_info=None,
    ),
    mdd = dict(
        type="MDD",
        level=level,
        symbol_info=None,
    ),
    cr = dict(
        type="CR",
        level=level,
        symbol_info=None,
    ),
    sor = dict(
        type="SOR",
        level=level,
        symbol_info=None,
    ),
    vol = dict(
        type="VOL",
        level=level,
        symbol_info=None,
    )
)

embed_config = dict(
    type="TradingPatchEmbed",
    dense_input_dim=dense_input_dim,
    sparse_input_dim=sparse_input_dim,
    latent_dim=embedding_dim,
    output_dim=embedding_dim,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    patch_timestamps=patch_timestamps,
    history_timestamps=history_timestamps,
    level=level,
    if_use_sparse=if_use_sparse,
    if_use_trajectory=if_use_trajectory,
    dropout=dropout
)

encoder_config = dict(
    type="TransformerEncoder",
    input_dim=embedding_dim,
    latent_dim=embedding_dim,
    output_dim=embedding_dim,
    depth=depth,
    num_heads=num_heads,
)

agent = dict(
    type="SAC",
    task_type=task_type,
    embed_config=embed_config,
    encoder_config=encoder_config,
    action_dim=action_dim,
    actor_output_dim=actor_output_dim,
    critic_output_dim=critic_output_dim,
)

trainer = dict(
    type="SACTradingTrainer",
    config = None,
    dataset=None,
    metrics = None,
    agent = None,
    device = None,
    dtype = None,
)

task = dict(
    type="Task",
    trainer=None,
    train=None,
    test=None,
    task_type=task_type
)