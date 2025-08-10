workdir = "workdir"
tag = "pretrain_day_dj30_factorvae"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"
tensorboard_path = "tensorboard"
checkpoint_path = "checkpoint"
wandb_path = "wandb"
resume = None
history_timestamps = 64
num_assets = 29
future_timestamps = 32
patch_timestamps = 64
feature_dim = 152
temporal_dim = 3
factor_num = 8
portfolio_num = 10
num_workers = 8
start_epoch = 0
timestamp_format = "%Y-%m-%d"
if_norm = True
if_norm_temporal = True
if_mask = False
if_use_future = False

# params
seed = 1337
cont_loss_factor = 0.1
batch_size = 32

# optimizer
vae_lr = 2e-5
vae_min_lr = 0.0
vae_weight_decay = 0.05
vae_betas = (0.9, 0.95)
dit_lr = 2e-5
dit_min_lr = 0.0
dit_weight_decay = 0.05
dit_betas = (0.9, 0.95)

# scheduler
num_training_epochs = int(500)
num_training_warmup_epochs = int(50)
num_checkpoint_del = 10
checkpoint_period = 20
repeat_aug = 1
num_training_data = int(1e6)
num_training_steps = int(1e6)
num_training_steps_per_epoch = int(1e3)
num_training_warmup_steps = int(1e2)

encoder_embed_dim = 128

dit_embed_dim = 128
dit_depth = 24
dit_num_heads = 16
dit_mlp_ratio = 4.0

decoder_embed_dim = 128

data_size = (history_timestamps, num_assets, feature_dim)
patch_size = (patch_timestamps, 1, feature_dim)
input_channel = 1
grad_clip = 1.0
dtype = "fp32"
num_classes = 3,
dropout_prob = 0.1

kl_loss_weight = 1e-2
nll_loss_weight = 1.0
cont_loss_weight = 0.1

num_plot_samples = 10
num_plot_samples_per_batch = 1 # num_plot_sample_batch = num_plot_samples // num_plot_samples_per_batch
num_plot_samples_asset_in_per_batch = 1

dataset = dict(
    type="MultiAssetPriceTextDataset",
    data_path="datasets/processd_day_dj30/features",
    assets_path="configs/_asset_list_/dj30.json",
    fields_name={
        "features": [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "kmid",
            "kmid2",
            "klen",
            "kup",
            "kup2",
            "klow",
            "klow2",
            "ksft",
            "ksft2",
            "roc_5",
            "roc_10",
            "roc_20",
            "roc_30",
            "roc_60",
            "ma_5",
            "ma_10",
            "ma_20",
            "ma_30",
            "ma_60",
            "std_5",
            "std_10",
            "std_20",
            "std_30",
            "std_60",
            "beta_5",
            "beta_10",
            "beta_20",
            "beta_30",
            "beta_60",
            "max_5",
            "max_10",
            "max_20",
            "max_30",
            "max_60",
            "min_5",
            "min_10",
            "min_20",
            "min_30",
            "min_60",
            "qtlu_5",
            "qtlu_10",
            "qtlu_20",
            "qtlu_30",
            "qtlu_60",
            "qtld_5",
            "qtld_10",
            "qtld_20",
            "qtld_30",
            "qtld_60",
            "rank_5",
            "rank_10",
            "rank_20",
            "rank_30",
            "rank_60",
            "imax_5",
            "imax_10",
            "imax_20",
            "imax_30",
            "imax_60",
            "imin_5",
            "imin_10",
            "imin_20",
            "imin_30",
            "imin_60",
            "imxd_5",
            "imxd_10",
            "imxd_20",
            "imxd_30",
            "imxd_60",
            "rsv_5",
            "rsv_10",
            "rsv_20",
            "rsv_30",
            "rsv_60",
            "cntp_5",
            "cntp_10",
            "cntp_20",
            "cntp_30",
            "cntp_60",
            "cntn_5",
            "cntn_10",
            "cntn_20",
            "cntn_30",
            "cntn_60",
            "cntd_5",
            "cntd_10",
            "cntd_20",
            "cntd_30",
            "cntd_60",
            "corr_5",
            "corr_10",
            "corr_20",
            "corr_30",
            "corr_60",
            "cord_5",
            "cord_10",
            "cord_20",
            "cord_30",
            "cord_60",
            "sump_5",
            "sump_10",
            "sump_20",
            "sump_30",
            "sump_60",
            "sumn_5",
            "sumn_10",
            "sumn_20",
            "sumn_30",
            "sumn_60",
            "sumd_5",
            "sumd_10",
            "sumd_20",
            "sumd_30",
            "sumd_60",
            "vma_5",
            "vma_10",
            "vma_20",
            "vma_30",
            "vma_60",
            "vstd_5",
            "vstd_10",
            "vstd_20",
            "vstd_30",
            "vstd_60",
            "wvma_5",
            "wvma_10",
            "wvma_20",
            "wvma_30",
            "wvma_60",
            "vsump_5",
            "vsump_10",
            "vsump_20",
            "vsump_30",
            "vsump_60",
            "vsumn_5",
            "vsumn_10",
            "vsumn_20",
            "vsumn_30",
            "vsumn_60",
            "vsumd_5",
            "vsumd_10",
            "vsumd_20",
            "vsumd_30",
            "vsumd_60",
        ],
        "prices": [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
        ],
        "temporals": [
            "day",
            "weekday",
            "month",
        ],
        "labels": [
            "ret1",
            "mov1"
        ]
    },
    if_norm=if_norm,
    if_norm_temporal=if_norm_temporal,
    if_use_future=if_use_future,
    scaler_cfg = dict(type="WindowedScaler"),
    scaler_file="scalers.joblib",
    scaled_data_file="scaled_data.joblib",
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    # start_timestamp="1994-03-01",
    start_timestamp="2008-04-01",
    end_timestamp="2024-04-01",
    timestamp_format = "%Y-%m-%d",
    exp_path=exp_path,
)

train_dataset = dataset.copy()
train_dataset.update(
    # start_timestamp="1994-03-01",
    scaler_file="train_scalers.joblib",
    scaled_data_file="train_scaled_data.joblib",
    start_timestamp="2008-04-01",
    end_timestamp="2021-04-01",
)

valid_dataset = dataset.copy()
valid_dataset.update(
    scaler_file="valid_scalers.joblib",
    scaled_data_file="valid_scaled_data.joblib",
    start_timestamp="2021-04-01",
    end_timestamp="2024-04-01",
)

test_dataset = dataset.copy()
test_dataset.update(
    scaler_file="test_scalers.joblib",
    scaled_data_file="test_scaled_data.joblib",
    start_timestamp="2021-04-01",
    end_timestamp="2024-04-01",
)

collate_fn = dict(
    type="MultiAssetPriceTextCollateFn"
)

embed_config = dict(
    type='FactorVAEEmbed',
    data_size=data_size,
    patch_size=patch_size,
    input_channel=input_channel,
    input_dim=feature_dim,
    embed_dim=encoder_embed_dim,
    temporal_dim=temporal_dim,
)

encoder_config = dict(
    type='FactorVAEEncoder',
    embed_config=embed_config,
    input_dim=encoder_embed_dim,
    latent_dim=encoder_embed_dim,
    factor_num=factor_num,
    portfolio_num=portfolio_num
)

decoder_config = dict(
    type='FactorVAEDecoder',
    embed_config=embed_config,
    input_dim=decoder_embed_dim,
    latent_dim=decoder_embed_dim,
    factor_num=factor_num,
    portfolio_num=portfolio_num
)

predictor_config = dict(
    type='FactorVAEPredictor',
    input_dim=encoder_embed_dim,
    latent_dim=encoder_embed_dim,
    factor_num=factor_num,
    portfolio_num=portfolio_num,
)

vae = dict(
    type = "FactorVAE",
    embed_config = embed_config,
    encoder_config = encoder_config,
    decoder_config = decoder_config,
    predictor_config = predictor_config
)

timestep_embed_config = dict(
    type="TimestepEmbed",
    embed_dim=dit_embed_dim,
    frequency_embedding_size=dit_embed_dim * 2,
)
label_embed_config = dict(
    type="LabelEmbed",
    embed_dim=dit_embed_dim,
    num_classes=num_classes,
    dropout_prob=dropout_prob
)
text_encoder_config = dict(
    type= "OpenAITextEncoder",
    provider_cfg_path="configs/openai_config.json",
    if_reduce_dim=True,
    reduced_dim=dit_embed_dim,
)

dit = dict(
    type="DiT",
    embed_config=embed_config,
    timestep_embed_config=timestep_embed_config,
    label_embed_config=label_embed_config,
    text_encoder_config=text_encoder_config,
    if_label_embed=False,
    if_text_encoder=True,
    input_dim=encoder_embed_dim,
    latent_dim=dit_embed_dim,
    output_dim=dit_embed_dim * 2,
    depth=dit_depth,
    num_heads=dit_num_heads,
    mlp_ratio=dit_mlp_ratio,
    cls_embed=True,
    sep_pos_embed=True,
    trunc_init=False,
    no_qkv_bias=False,
)

diffusion = dict(
    type="SpacedDiffusion",
    timestep_respacing=""
)

vae_optimizer = dict(
    type="AdamW",
    lr=vae_lr,
    weight_decay=vae_weight_decay,
    betas = vae_betas,
)

dit_optimizer = dict(
    type="AdamW",
    lr=dit_lr,
    weight_decay=dit_weight_decay,
    betas = dit_betas,
)

vae_scheduler = dict(
    type="LinearWithWarmupScheduler",
    num_warmup_steps=num_training_warmup_steps,
    num_training_steps=num_training_steps,
)

dit_scheduler = dict(
    type="LinearWithWarmupScheduler",
    num_warmup_steps=num_training_warmup_steps,
    num_training_steps=num_training_steps,
)



loss_funcs_config = dict(
    vae_loss=dict(
        type="FactorVAELoss",
        kl_loss_weight=kl_loss_weight,
        nll_loss_weight=nll_loss_weight,
    )
)

plot = dict(
    type="PlotInterface",
    sample_num = num_plot_samples_per_batch,
    sample_asset = num_plot_samples_asset_in_per_batch,
    suffix = 'jpeg'
)

trainer = dict(
    type = "FactorVAETrainer",
    vae = None,
    vae_ema = None,
    dit = None,
    dit_ema = None,
    diffusion = None,
    train_dataloader = None,
    valid_dataloader = None,
    vae_loss_fn = None,
    price_cont_loss_fn = None,
    vae_optimizer = None,
    dit_optimizer = None,
    vae_scheduler = None,
    dit_scheduler = None,
    logger = None,
    device = None,
    dtype = None,
    writer = None,
    wandb = None,
    num_plot_samples = num_plot_samples,
    plot = None,
    accelerator = None,
    exp_path = exp_path
)