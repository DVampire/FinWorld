task_type = "trading"
symbol = "GOOGL"
workdir = "workdir"
method = "macd"
tag = f"{symbol}_{method}_{task_type}"
exp_path = f"{workdir}/{tag}"
project = "rule"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

############Dataset Parameters############
# dataset parameters
level = "1day"
history_timestamps = 64
future_timestamps = 32
start_timestamp = "2015-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2023-05-01"
if_norm = True
if_use_temporal = True
if_norm_temporal = True
if_use_future = False

############Environment Parameters############
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99

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
    level=level
)

environment = dict(
    type="EnvironmentGeneralTrading",
    mode="train",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    gamma=gamma
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

agent = dict(
    type="MACD",
    task_type = task_type,
    level = level,
)

trainer = dict(
    type="RuleTrdingTrainer",
    config=None,
    dataset=None,
    agent=None,
    metrics=None,
)

task = dict(
    type="Task",
    trainer=None,
    train=None,
    test=None,
    task_type=task_type
)