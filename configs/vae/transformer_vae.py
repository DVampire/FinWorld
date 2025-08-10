symbol = "AAPL"
workdir = "workdir"
method = "transformer"
tag = f"{symbol}_vae_{method}"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"
tensorboard_path = "tensorboard"
checkpoint_path = "checkpoint"
wandb_path = "wandb"
plot_path = "plot"
project = "vae"

seed = 1337
batch_size = 32
history_timestamps = 64
future_timestamps = 32
patch_timestamps = 4
start_timestamp = "2015-05-01"
end_timestamp = "2025-05-01"
split_timestamp = "2023-05-01"
num_features = 154
if_norm = True
if_use_temporal = True
if_norm_temporal = True
if_use_future = False
data_size = (history_timestamps, num_features)
patch_size = (patch_timestamps, num_features)
device = "cuda"
num_workers = 4
pin_memory = True
distributed = True

# optimizer
vae_lr = 1e-4
vae_min_lr = 0.0
vae_weight_decay = 0.05
vae_betas = (0.9, 0.95)

# scheduler
num_training_epochs = int(1000)
num_training_warmup_epochs = int(100)
num_checkpoint_del = 10
checkpoint_period = 20
repeat_aug = 1
num_training_data = int(1e6)
num_training_steps = int(1e6)
num_training_steps_per_epoch = int(1e3)
num_training_warmup_steps = int(1e2)

# loss parameters
cont_loss_weight = 0.1
kl_loss_weight = 0.1
nll_loss_weight = 1e-3

# model parameters
encoder_embed_dim = 256
encoder_depth = 4
encoder_num_heads = 4
encoder_mlp_ratio = 4.0
decoder_embed_dim = 256
decoder_depth = 2
decoder_num_heads = 4
decoder_mlp_ratio = 4.0
pred_dim = 4 # 'close', 'high', 'low', 'open'
if_mask = False
mask_ratio_min = 0.4
mask_ratio_max = 0.8
mask_ratio_mu = 0.55
mask_ratio_std = 0.25
input_channel = 1
dtype = "fp32"

# plotting parameters
num_plot_samples = 4
num_plot_samples_per_batch = 1 # num_plot_sample_batch = num_plot_samples // num_plot_samples_per_batch
print_freq = 20

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
)

dataloader_config = dict(
    type="DataLoader",
    accelerator=None,
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
train_dataloader = dataloader_config.copy()
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
valid_dataloader = dataloader_config.copy()
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
test_dataloader = dataloader_config.copy()
test_dataloader.update(
    shuffle=False,
    drop_last=False,
    train=False,
)

collate_fn = dict(
    type="SingleAssetPriceTextCollateFn"
)

embed_config = dict(
    type='PatchEmbed',
    data_size=data_size,
    patch_size=patch_size,
    input_channel=input_channel,
    input_dim=num_features,
    latent_dim=encoder_embed_dim,
    output_dim=encoder_embed_dim,
)

encoder_config = dict(
        type='TransformerEncoder',
        input_dim=encoder_embed_dim,
        latent_dim=encoder_embed_dim,
        output_dim=encoder_embed_dim * 2,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
        if_mask=if_mask,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
        mask_ratio_mu=mask_ratio_mu,
        mask_ratio_std=mask_ratio_std,
    )

decoder_config = dict(
        type='TransformerDecoder',
        input_dim=encoder_embed_dim,
        latent_dim=decoder_embed_dim,
        output_dim=decoder_embed_dim,
        depth= decoder_depth,
        num_heads = decoder_num_heads,
        mlp_ratio= decoder_mlp_ratio,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
    )

vae = dict(
    type = "TransformerVAE",
    embed_config=embed_config,
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    output_dim = pred_dim,
)

vae_optimizer = dict(
    type="AdamW",
    lr= vae_lr,
    weight_decay= vae_weight_decay,
    betas = vae_betas,
)

vae_scheduler = dict(
    type="LinearWithWarmupScheduler",
    num_warmup_steps=num_training_warmup_steps,
    num_training_steps=num_training_steps,
)

loss_funcs_config = dict(
    vae_loss=dict(
        type="VAELoss",
        nll_loss_weight=nll_loss_weight,
        kl_loss_weight=kl_loss_weight,
    ),
    price_cont_loss = dict(
        type="PriceConstraintEntropyLoss",
        cont_loss_weight=cont_loss_weight,
    )
)

plot = dict(
    type="PlotKline",
)

trainer = dict(
    type = "VAETrainer",
    config = None,
    vae = None,
    vae_ema = None,
    train_dataloader = None,
    valid_dataloader = None,
    loss_funcs = None,
    vae_optimizer = None,
    vae_scheduler = None,
    dit_scheduler = None,
    device = None,
    dtype = None,
    writer = None,
    wandb = None,
    plot = None,
    accelerator = None,
)
