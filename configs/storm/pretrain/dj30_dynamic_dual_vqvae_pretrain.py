task_type = "pretrain"
assets_name = "dj30"
workdir = "workdir"
method = "dynamic_dual_vqvae"
tag = f"{assets_name}_{method}_{task_type}"
exp_path = f"{workdir}/{tag}"
project = "storm"
log_path = "finworld.log"
checkpoint_path = "checkpoint"
plot_path = "plot"

#-----------------Dataset Parameters-----------------#
# dataset parameters
history_timestamps = 64
review_timestamps = 32
future_timestamps = 32
patch_timestamps = 4
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
num_workers = 1
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
encoder_embed_dim = embedding_dim
encoder_depth = 4
encoder_num_heads = 4
encoder_mlp_ratio = 4.0
cs_codebook_size =  512
cs_codebook_dim = encoder_embed_dim
ts_codebook_size =  512
ts_codebook_dim = encoder_embed_dim
decoder_embed_dim = embedding_dim
decoder_depth = 2
decoder_num_heads = 4
decoder_mlp_ratio = 4.0
output_dim = 4 # close, high, low, open
input_channel = 1
if_use_multi_scale_encoder = False
multi_scale_encoder_depth = 2
multi_scale_encoder_heads = 4
multi_scale_encoder_dim_head = 8
if_mask = False
mask_ratio_min = 0.4
mask_ratio_max = 0.8
mask_ratio_mu = 0.55
mask_ratio_std = 0.25
cs_data_size = (history_timestamps, num_assets, dense_input_dim)
cs_patch_size = (1, num_assets, dense_input_dim)
ts_data_size = (history_timestamps, num_assets, dense_input_dim)
ts_patch_size = (patch_timestamps, 1, dense_input_dim)
temperature = 1.0

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

#-----------------Loss Parameters-----------------#
cs_scale = 1e-3
cl_loss_weight = 1e-3
ret_loss_weight = 0.1
nll_loss_weight = 1e-3
cont_loss_weight = 0.1
commitment_loss_weight = 1.0
kl_loss_weight = 0.1
orthogonal_reg_loss_weight = 0.1
codebook_diversity_loss_weight = 0.1
orthogonal_reg_max_codes = int(1024 // 2)
orthogonal_reg_active_codes_only = False
codebook_diversity_temperature = 1.0

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

cs_embed_config = dict(
    type='PatchEmbed',
    data_size=cs_data_size,
    patch_size=cs_patch_size,
    input_channel=input_channel,
    input_dim=dense_input_dim,
    latent_dim=encoder_embed_dim,
    output_dim=encoder_embed_dim,
)

ts_embed_config = dict(
    type='PatchEmbed',
    data_size=ts_data_size,
    patch_size=ts_patch_size,
    input_channel=input_channel,
    input_dim=dense_input_dim,
    latent_dim=encoder_embed_dim,
    output_dim=encoder_embed_dim,
)

cs_config = dict(
        cs_encoder_config = dict(
            type = "TransformerEncoder",
            embed_config=cs_embed_config,
            input_dim=encoder_embed_dim,
            latent_dim=encoder_embed_dim,
            output_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
            if_mask=if_mask,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            mask_ratio_mu=mask_ratio_mu,
            mask_ratio_std=mask_ratio_std,
        ),
        cs_quantizer_config = dict(
            type="VectorQuantizer",
            dim=cs_codebook_dim,
            codebook_size=cs_codebook_size,
            codebook_dim=cs_codebook_dim,
            decay=0.99,
            commitment_weight=commitment_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_loss_weight,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            codebook_diversity_temperature=codebook_diversity_temperature
        ),
        cs_decoder_config = dict(
            type='TransformerDecoder',
            embed_config=cs_embed_config,
            input_dim=encoder_embed_dim,
            latent_dim=decoder_embed_dim,
            output_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
        )
    )

ts_config = dict(
        ts_encoder_config = dict(
            type = "TransformerEncoder",
            embed_config=ts_embed_config,
            input_dim=encoder_embed_dim,
            latent_dim=encoder_embed_dim,
            output_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
            if_mask=if_mask,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            mask_ratio_mu=mask_ratio_mu,
            mask_ratio_std=mask_ratio_std,
        ),
        ts_quantizer_config = dict(
            type="VectorQuantizer",
            dim=ts_codebook_dim,
            codebook_size=ts_codebook_size,
            codebook_dim=ts_codebook_dim,
            decay=0.99,
            commitment_weight=commitment_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_loss_weight,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            codebook_diversity_temperature=codebook_diversity_temperature
        ),
        ts_decoder_config = dict(
            type='TransformerDecoder',
            embed_config=ts_embed_config,
            input_dim=encoder_embed_dim,
            latent_dim=decoder_embed_dim,
            output_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
        )
    )

enc_params = dict(
    depth=multi_scale_encoder_depth,
    heads=multi_scale_encoder_heads,
    mlp_dim=encoder_embed_dim,
    dim_head=multi_scale_encoder_dim_head
)

multi_scale_encoder_config = dict(
    depth=multi_scale_encoder_depth,
    sm_dim=encoder_embed_dim,
    lg_dim=encoder_embed_dim,
    cross_attn_depth=multi_scale_encoder_depth,
    cross_attn_heads=multi_scale_encoder_heads,
    cross_attn_dim_head=multi_scale_encoder_dim_head,
    sm_enc_params=enc_params,
    lg_enc_params=enc_params
)

model = dict(
    type = "DynamicDualVQVAE",
    cs_embed_config=cs_embed_config,
    ts_embed_config=ts_embed_config,
    cs_config=cs_config,
    ts_config=ts_config,
    multi_scale_encoder_config=multi_scale_encoder_config,
    if_use_multi_scale_encoder=if_use_multi_scale_encoder,
    cl_loss_weight=cl_loss_weight,
    temperature=temperature,
    asset_num=num_assets,
    output_dim=output_dim,
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

metric = dict(
    mae = dict(
        type="MAE",
    ),
    mse = dict(
        type="MSE",
    ),
    rankic = dict(
        type="RANKIC",
    ),
    rankicir = dict(
        type="RANKICIR",
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

loss = dict(
    vae_loss=dict(
        type="DualVQVAELoss",
        cs_scale=cs_scale,
        nll_loss_weight=nll_loss_weight,
        ret_loss_weight=ret_loss_weight,
        kl_loss_weight=kl_loss_weight,
    ),
    price_loss = dict(
        type="PriceLoss",
        loss_weight=cont_loss_weight,
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
    type="DynamicDualVQVAETrainer",
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