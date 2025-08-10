task_type = "portfolio"
assets_name = "dj30"
workdir = "workdir"
method = "lightgbm"
tag = f"{assets_name}_{method}_{task_type}"
exp_path = f"{workdir}/{tag}"
project = "ml"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

#-----------------Dataset Parameters-----------------#
# dataset parameters
exclude_assets = []
level = "1day"
history_timestamps = 64
future_timestamps = 32
start_timestamp = "2015-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2023-05-01"
num_assets = 29 -len(exclude_assets)  # = 29 - 0 = 29
if_norm = True
if_use_temporal = True
if_norm_temporal = True
if_use_future = False

#----------------Environment Parameters------------#
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99

#-----------------Model Parameters-----------------#
seed = 1024
n_estimators = 1000
learning_rate = 0.05
num_leaves = 31
max_depth = -1
min_child_samples = 50
subsample = 0.8
subsample_freq = 1
colsample_bytree = 0.8
reg_lambda = 1.0
reg_alpha = 0.0
random_state = seed
n_jobs = 1
n_components = 32  # Number of components for PCA

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
    type="MultiAssetDataset",
    assets_name=assets_name,
    exclude_assets=exclude_assets,
    data_path=f"datasets/{assets_name}",
    enabled_data_configs=[
        {
            "asset_name": assets_name,
            "source": "fmp",
            "data_type": "price",
            "level": "1day",
        },
        {
            "asset_name": assets_name,
            "source": "fmp",
            "data_type": "feature",
            "level": "1day",
        },
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
    type="EnvironmentGeneralPortfolio",
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
    type="Lightgbm",
    task_type = task_type,
    level = level,
    num_regressors=num_assets,
    params=dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=random_state,
        n_jobs=n_jobs
    ),
    reducer = dict(
        type="PCA",
        n_components=n_components,
    )
)

downstream_environment = dict(
    type="EnvironmentGeneralPortfolio",
    mode="valid",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    gamma=gamma
)

downstream = dict(
    type="TopkDropoutStrategy",
)

trainer = dict(
    type="MLPortfolioTrainer",
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