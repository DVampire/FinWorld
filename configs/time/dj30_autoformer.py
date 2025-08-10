task_type = "forecasting"
assets_name = "dj30"
workdir = "workdir"
method = "autoformer"
tag = f"{assets_name}_{method}_{task_type}"
exp_path = f"{workdir}/{tag}"
project = "time"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

#-----------------Dataset Parameters-----------------#
# dataset parameters
history_timestamps = 64
review_timestamps = 32
future_timestamps = 32
num_assets = 29 # 29 stocks in DJ30
start_timestamp = "2015-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2023-05-01"
if_norm = True
if_use_temporal = True
if_norm_temporal = False
if_use_future = True
if_use_news = False
level = "1day"
# dataloader parameters
batch_size = 64 # Batch size
num_workers = 4
pin_memory = True
distributed = True

#----------------Environment Parameters------------#
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99

#-----------------Model Parameters-----------------#
dense_input_dim = 150
sparse_input_dim = 4 # days, months, weekdays, years
embedding_dim = 128
dropout = 0.1
moving_avg = 25
factor = 1
input_length = history_timestamps
output_length = future_timestamps
encode_input_features = dense_input_dim + sparse_input_dim
decode_input_features = dense_input_dim + sparse_input_dim
output_dim = num_assets
# Encoder and Decoder parameters
encoder_input_dim = embedding_dim
encoder_latent_dim = embedding_dim
encoder_output_dim = embedding_dim
encoder_depth = 4
encoder_mlp_ratio = 4.0
encoder_num_heads = 4
decoder_input_dim = embedding_dim
decoder_latent_dim = embedding_dim
decoder_output_dim = embedding_dim
decoder_depth = encoder_depth
decoder_mlp_ratio = 4.0
decoder_num_heads = 4

#-----------------Optimizer Parameters-----------------#
weight_decay = 0.05
betas = (0.9, 0.95)
lr = 1e-5

#-----------------Scheduler Parameters-----------------#
num_training_epochs = int(2000) # Training epochs
num_training_warmup_epochs = int(200) # Warmup epochs

# Will be calculated later
num_device = None
num_training_data = None
num_training_steps = None
num_training_steps_per_epoch = None
num_training_warmup_steps = None

#-----------------Trainer Parameters-----------------#
checkpoint_period = 10 # Checkpoint period
num_checkpoint_del = 5 # Number of checkpoints to delete
print_freq = 10 # Print frequency
num_plot_samples = 4 # Number of samples to plot
num_plot_samples_per_batch = 1 # Number of samples per batch to plot
num_plot_sample_batch = int(num_plot_samples / num_plot_samples_per_batch)
device = "cuda"
dtype = "fp32"
fp32 = True if dtype == "fp32" else False
clip_grad = 0.02

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

# Dataset configuration
dataset = dict(
    type="MultiAssetDataset",
    assets_name=assets_name,
    data_path=f"datasets/{assets_name}",
    enabled_data_configs = [
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
    if_use_news=if_use_news,
    scaler_cfg = dict(
        type="WindowedScaler"
    ),
    history_timestamps = history_timestamps,
    future_timestamps = future_timestamps,
    start_timestamp= start_timestamp,
    end_timestamp= end_timestamp,
    level=level
)

dataloader = dict(
    type="DataLoader",
    collate_fn=None,
    dataset=None,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=pin_memory,
    distributed=distributed,
    train=True,
)

train_dataset = dataset.copy()
train_dataset.update(
    start_timestamp=start_timestamp,
    end_timestamp=split_timestamp,
)
train_dataloader = dataloader.copy()
train_dataloader.update(
    shuffle=True,
    drop_last=True,
    train=True,
)
valid_dataset = dataset.copy()
valid_dataset.update(
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp
)
valid_dataloader = dataloader.copy()
valid_dataloader.update(
    shuffle=False,
    drop_last=False,
    train=False,
)
test_dataset = dataset.copy()
test_dataset.update(
    start_timestamp= split_timestamp,
    end_timestamp=end_timestamp
)
test_dataloader = dataloader.copy()
test_dataloader.update(
    shuffle=False,
    drop_last=False,
    train=False,
)

collate_fn = dict(
    type="MultiAssetPriceTextCollateFn"
)

encoder_embed_config = dict(
    type="AggDataEmbed",
    dense_input_dim=dense_input_dim,
    sparse_input_dim=sparse_input_dim,
    latent_dim=embedding_dim,
    output_dim=embedding_dim,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    level=level,
    dropout=dropout
)

decoder_embed_config = dict(
    type="AggDataEmbed",
    dense_input_dim=dense_input_dim,
    sparse_input_dim=sparse_input_dim,
    latent_dim=embedding_dim,
    output_dim=embedding_dim,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    level=level,
    dropout=dropout
)

encoder_config = dict(
    type="AutoformerEncoder",
    input_dim=encoder_input_dim,
    latent_dim=encoder_latent_dim,
    output_dim=encoder_output_dim,
    depth=encoder_depth,
    num_heads=encoder_num_heads,
    mlp_ratio=encoder_mlp_ratio,
    no_qkv_bias=False,
    moving_avg=moving_avg,
    factor=factor
)

decoder_config = dict(
    type="AutoformerDecoder",
    input_dim=decoder_input_dim,
    latent_dim=decoder_latent_dim,
    output_dim=decoder_output_dim,
    depth=decoder_depth,
    num_heads=decoder_num_heads,
    mlp_ratio=decoder_mlp_ratio,
    no_qkv_bias=False,
    moving_avg=moving_avg,
    factor=factor
)

model = dict(
    type="Autoformer",
    task_type=task_type,
    encoder_embed_config=encoder_embed_config,
    decoder_embed_config=decoder_embed_config,
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    review_timestamps=review_timestamps,
    moving_avg=moving_avg,
    output_dim=num_assets
)

optimizer = dict(
    type="AdamW",
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
)

scheduler = dict(
    type="LinearWithWarmupScheduler",
    num_warmup_steps=None,
    num_training_steps=None,
)

loss = dict(
    mse = dict(
        type="MSELoss",
        reduction="mean",
        loss_weight=1.0
    )
)

metric = dict(
    mae = dict(
        type="MAE",
    ),
    mse = dict(
        type="MSE",
    ),
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
    type="ForecastingTrainer",
    config=None,
    model=None,
    dataset=None,
    optimizer=None,
    scheduler=None,
    losses=None,
    metrics=None,
    plot=None,
    device=None,
    dtype=None,
)

task = dict(
    type="Task",
    trainer=None,
    train=None,
    test=None,
    task_type=task_type
)