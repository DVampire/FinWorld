import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from torch.distributions.categorical import Categorical
from typing import Tuple
from tensordict import TensorDict
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Dirichlet
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from finworld.registry import AGENT
from finworld.models.rl.actor import Actor
from finworld.models.rl.critic import Critic
from finworld.task import TaskType


@AGENT.register_module(force=True)
class PPO(nn.Module):
    def __init__(self,
                 *args,
                 task_type: str,
                 embed_config: dict,
                 encoder_config: dict,
                 action_dim: int,
                 actor_output_dim: int,
                 critic_output_dim: int,
                 **kwargs
                 ):

        super(PPO, self).__init__()
        
        self.task_type = TaskType.from_string(task_type)
        self.action_dim = action_dim
        self.actor_output_dim = actor_output_dim
        self.critic_output_dim = critic_output_dim

        self.actor = Actor(
            task_type=task_type,
            embed_config=embed_config,
            encoder_config=encoder_config,
            action_dim=action_dim,
            output_dim=actor_output_dim
        )
        
        self.critic = Critic(
            task_type=task_type,
            embed_config=embed_config,
            encoder_config=encoder_config,
            action_dim=action_dim,
            output_dim=critic_output_dim
        )

    def get_value(self, x: TensorDict, a: Tensor = None):
        return self.critic(x, a)

    def get_action(self, x: TensorDict):
        batch_size = x.batch_size

        if len(batch_size) == 2:
            x = TensorDict(
                {
                    key: rearrange(value, 'b e ... -> (b e) ...') for key, value in x.items()
                },
                batch_size=batch_size[0] * batch_size[1]
            )

        logits = self.actor(x)

        if self.task_type == TaskType.TRADING:
            dis = Categorical(logits=logits)
            action = dis.sample()
            return action
        elif self.task_type == TaskType.PORTFOLIO:
            alpha = F.softplus(logits) + 1e-3
            dist = Dirichlet(alpha)
            action = dist.rsample()
            return action

    def get_action_and_value(self, x: TensorDict, a: Tensor = None):
        batch_size = x.batch_size

        if len(batch_size) == 2:
            x = TensorDict(
                {
                    key: rearrange(value, 'b e ... -> (b e) ...') for key, value in x.items()
                },
                batch_size=batch_size[0] * batch_size[1]
            )
            
            if a is not None:
                a = rearrange(a, 'b e ... -> (b e) ...')

        logits = self.actor(x)
        
        if self.task_type == TaskType.TRADING:
            
            dis = Categorical(logits=logits)

            if a is None:
                a = dis.sample()
            else:
                a = a[..., -1].view(-1)

            probs = dis.log_prob(a)

            entropy = dis.entropy()
            
            value = self.critic(x, a)
            
            if len(batch_size) == 2:
                action = rearrange(a, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
                probs = rearrange(probs, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
                entropy = rearrange(entropy, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
                value = rearrange(value, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
            else:
                action = a
                probs = probs
                entropy = entropy
                value = value
                
        elif self.task_type == TaskType.PORTFOLIO:
            
            alpha = F.softplus(logits) + 1e-3
            
            dist = Dirichlet(alpha)
            
            if a is None:
                a = dist.rsample()
                
            probs = dist.log_prob(a)
            
            entropy = dist.entropy()
            
            value = self.critic(x, a)
            
            if len(batch_size) == 2:
                action = rearrange(a, "(b c) n -> b c n", b=batch_size[0], c=batch_size[1])
                probs = rearrange(probs, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
                entropy = rearrange(entropy, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
                value = rearrange(value, "(b c) -> b c", b=batch_size[0], c=batch_size[1])
            else:
                action = a
                probs = probs
                entropy = entropy
                value = value
                
        return action, probs, entropy, value

    def forward(self, *input, **kwargs):
        pass

if __name__ == '__main__':
    device = torch.device("cpu")
    
    task_type = "trading"
    batch_size = 4
    num_envs = 4
    seq_len = 64
    num_assets = 6
    action_dim = 3
    dense_input_dim = 64
    sparse_input_dim = 4
    latent_dim = 64
    output_dim = 64
    actor_output_dim = 3
    critic_output_dim = 1
    
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
    
    ppo = PPO(
        task_type=task_type,
        embed_config=embed_config,
        encoder_config=encoder_config,
        action_dim=action_dim,
        actor_output_dim=actor_output_dim,
        critic_output_dim=critic_output_dim
    )

    dense_features = torch.randn(batch_size, seq_len, 64)  # Batch size of 4, 10 time steps, 64 features
    years = torch.randint(0, 10, (batch_size, seq_len, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, seq_len, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, seq_len, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, seq_len, 1))  # Day feature
    sparse_features = torch.cat([days, months, weekdays, years], dim=-1).to(
        device)  # Shape: (batch_size, seq_len, num_features)
    x = TensorDict({
        'dense': dense_features.to(device),
        'sparse': sparse_features.to(device),
        "cashes": torch.randn(batch_size, seq_len).to(device),
        "positions": torch.randn(batch_size, seq_len).to(device),
        "rets": torch.randn(batch_size, seq_len).to(device),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len)).to(device)
    }, batch_size=(batch_size,)).to(device)
    output = ppo.get_value(x)
    print("Output shape:", output.shape)
    action, probs, entropy, value = ppo.get_action_and_value(x)
    print("Action shape:", action.shape)
    print("Probs shape:", probs.shape)
    print("Entropy shape:", entropy.shape)
    print("Value shape:", value.shape)
    
    #### for portfolio task####
    task_type = "portfolio"
    action_dim = 6
    actor_output_dim = 6
    critic_output_dim = 1

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
    
    ppo = PPO(
        task_type=task_type,
        embed_config=embed_config,
        encoder_config=encoder_config,
        action_dim=action_dim + 1, # 0 for cash
        actor_output_dim=actor_output_dim + 1, # 0 for cash
        critic_output_dim=critic_output_dim
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
    output = ppo.get_value(x)
    print("Output shape:", output.shape)
    action, probs, entropy, value = ppo.get_action_and_value(x)
    print("Action shape:", action.shape)
    print("Probs shape:", probs.shape)
    print("Entropy shape:", entropy.shape)
    print("Value shape:", value.shape)