task_type = "trading"
symbol = "AAPL"
workdir = "workdir"
method = "finance_agent"
tag = f"{symbol}_{method}_{task_type}"
exp_path = f"{workdir}/{tag}"
project = "agent"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

############Dataset Parameters############
# dataset parameters
level = "1day"
history_timestamps = 5
future_timestamps = 0
start_timestamp = "2015-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2023-05-01"
num_features = 154
if_norm = True
if_use_temporal = True
if_norm_temporal = True
if_use_future = False

############Environment Parameters############
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99
record_max_len=32
valid_action_max_len=8
single_text_max_tokens=1024
single_text_min_tokens=256
daily_sample_texts=2
max_steps = 20

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
    type="EnvironmentAgentTrading",
    mode="train",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    gamma=gamma,
    record_max_len=record_max_len,
    valid_action_max_len=valid_action_max_len,
    single_text_max_tokens=single_text_max_tokens,
    single_text_min_tokens=single_text_min_tokens,
    daily_sample_texts=daily_sample_texts,
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
    type="FinanceAgent",
    model="Qwen3-32B",
    name="finance_agent",
    description="This is a finance agent that generates actions based on financial data and news.",
    tools=[],
    max_steps=max_steps,
    provide_run_summary=True,
    template_path="finworld/agent/agent/finance_agent/prompts/finance_agent.yaml",
)

trainer = dict(
    type="FinanceAgentTrainer",
    config=None,
    train_environment=None,
    valid_environment=None,
    test_environment=None,
    agent=None,
    metrics=None,
)

task = dict(
    type="AsyncTask",
    trainer=None,
    train=None,
    test=None,
    task_type=task_type
)