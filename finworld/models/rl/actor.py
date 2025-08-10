import torch
import torch.nn as nn
from tensordict import TensorDict
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange

from finworld.registry import EMBED
from finworld.registry import ENCODER
from finworld.models.base import Model
from finworld.models.modules.attention_pool import AttentionPool1D
from finworld.models.modules.mean_pool import MeanPool1D
from finworld.task import TaskType
from finworld.models.embed.position import SinCosPosition2DEmbed, SinCosPosition1DEmbed

class Actor(Model):
    def __init__(self,
                 task_type: str,
                 embed_config: dict,
                 encoder_config: dict,
                 action_dim: int,
                 output_dim: int,
                 pool_type: str = 'avg', # 'avg' or 'attn'
                 **kwargs):
        super(Actor, self).__init__(**kwargs)
        
        self.task_type = TaskType.from_string(task_type)
        
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.embed = EMBED.build(embed_config)

        if self.task_type == TaskType.TRADING:
            self.pos_embed = SinCosPosition1DEmbed(
                num_positions=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
        elif self.task_type == TaskType.PORTFOLIO:
            self.pos_embed = SinCosPosition2DEmbed(
                num_time=embed_config["history_timestamps"] // embed_config["patch_timestamps"],
                num_space= embed_config["num_assets"],
                embed_dim=embed_config["output_dim"],
                num_prefix=0
            )
        self.encoder = ENCODER.build(encoder_config)
        self.pool_type = pool_type

        if self.pool_type == 'avg':
            self.pool = MeanPool1D(dim=1, keepdim=False)
        elif self.pool_type == 'attn':
            self.pool = AttentionPool1D(
                in_features=encoder_config['output_dim'],
                out_features=output_dim,
                embed_dim=encoder_config['latent_dim'],
                num_heads=encoder_config['num_heads'],
            )
        
        self.decoder = nn.Linear(encoder_config['output_dim'], output_dim)
        
        self.initialize_weights()
        
    @apply_forward_hook
    def encode(self, x: TensorDict):
        x = self.embed(x)

        pos = self.pos_embed(x)
        x = x + pos

        x, _, _ = self.encoder(x)
        return x
    
    @apply_forward_hook
    def decode(self, x: TensorDict):
        x = self.pool(x)
        x = self.decoder(x)
        return x
        
    def forward(self, x: TensorDict):
        batch_size = x.batch_size
        
        if len(batch_size) == 2:
            x = TensorDict(
                {
                    key: rearrange(value, 'b e ... -> (b e) ...') for key, value in x.items()
                },
                batch_size=batch_size[0] * batch_size[1]
            )
            
        x = self.encode(x)
        x = self.decode(x)
        
        if len(batch_size) == 2:
            x = rearrange(x, '(b e) ... -> b e ...', b=batch_size[0])
            
        return x
    
if __name__ == "__main__":
    device = torch.device('cpu')
    
    batch_size = 4
    num_envs = 4
    seq_len = 64
    num_assets = 6
    action_dim = 3
    dense_input_dim = 64
    sparse_input_dim = 4
    latent_dim = 64
    output_dim = 64

    
    #### for trading task####
    embed_config = dict(
        type="TradingPatchEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        patch_timestamps=4,
        history_timestamps=seq_len,
        level='1day',
        if_use_sparse=True,
        if_use_trajectory=True,
        dropout=0.1
    )
    
    encoder_config = dict(
        type="TransformerEncoder",
        input_dim=64,
        latent_dim=64,
        output_dim=64,
        depth=2,
        num_heads=4,
    )
    
    actor = Actor(
        task_type="trading",
        embed_config=embed_config,
        encoder_config=encoder_config,
        action_dim=action_dim,
        output_dim=action_dim
    )
    dense_features = torch.randn(batch_size, seq_len, 64)  # Batch size of 4, 10 time steps, 64 features
    years = torch.randint(0, 10, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature
    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)
    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device),
        "cashes": torch.randn(batch_size, seq_len).to(device),
        "positions": torch.randn(batch_size, seq_len).to(device),
        "rets": torch.randn(batch_size, seq_len).to(device),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len)).to(device)
    }, batch_size=(batch_size,)).to(device)
    output = actor(x)
    print("Output shape:", output.shape)
    
    dense_features = torch.randn(batch_size, num_envs, seq_len, 64)
    years = torch.randint(0, 10, (batch_size, num_envs, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, num_envs, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, num_envs, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, num_envs, seq_len, 1))  # Day feature
    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)
    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device),
        "cashes": torch.randn(batch_size, num_envs, seq_len).to(device),
        "positions": torch.randn(batch_size, num_envs, seq_len).to(device),
        "rets": torch.randn(batch_size, num_envs, seq_len).to(device),
        "actions": torch.randint(0, action_dim, (batch_size, num_envs, seq_len)).to(device)
    }, batch_size=(batch_size, num_envs)).to(device)
    output = actor(x)
    print("Output shape:", output.shape)
    
    #### for portfolio task####
    embed_config = dict(
        type="PortfolioPatchEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        num_assets=num_assets,
        latent_dim=latent_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        patch_timestamps=4,
        history_timestamps=seq_len,
        level='1day',
        if_use_sparse=True,
        if_use_trajectory=False,
        dropout=0.1
    )
    
    encoder_config = dict(
        type="TransformerEncoder",
        input_dim=64,
        latent_dim=64,
        output_dim=64,
        depth=2,
        num_heads=4,
    )
    
    dense_features = torch.randn(batch_size, seq_len, num_assets, 64)  # Batch size of 2, 10 time steps, num_asset assets, 64 features
    years = torch.randint(0, 10, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature
    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)
    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device)
    }, batch_size=(batch_size,)).to(device)
    
    actor = Actor(
        task_type="portfolio",
        embed_config=embed_config,
        encoder_config=encoder_config,
        action_dim=num_assets + 1, # 0 for cash
        output_dim=num_assets + 1 # 0 for cash
    )
    output = actor(x)
    print("Output shape:", output.shape)
    
    dense_features = torch.randn(batch_size, num_envs, seq_len, num_assets, 64)
    years = torch.randint(0, 10, (batch_size, num_envs, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, num_envs, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, num_envs, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, num_envs, seq_len, 1))  # Day feature
    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(device)  # Shape: (batch_size, seq_len, num_features)
    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device)
    }, batch_size=(batch_size, num_envs)).to(device)
    output = actor(x)
    print("Output shape:", output.shape)