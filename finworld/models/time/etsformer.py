import torch
from torch import nn
from typing import Dict, List
from tensordict import TensorDict
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange, repeat

from finworld.registry import EMBED
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import MODEL
from finworld.task import TaskType
from finworld.models.embed.position import AbsPosition2DEmbed

@MODEL.register_module(force=True)
class Etsformer(nn.Module):
    def __init__(self,
                 *args,
                 task_type: str = 'forecasting',
                 encoder_embed_config: Dict,
                 encoder_config: Dict,
                 decoder_config: Dict,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 review_timestamps: int = 32,
                 output_dim: int = 6,
                 **kwargs
                 ) -> None:
        super(Etsformer, self).__init__(*args, **kwargs)

        self.task_type = TaskType.from_string(task_type)

        self.encoder_embed_config = encoder_embed_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.encoder_embed = EMBED.build(encoder_embed_config)
        self.encoder = ENCODER.build(encoder_config)
        self.decoder = DECODER.build(decoder_config)

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.review_timestamps = review_timestamps  # Review timestamps from history
        self.output_dim = output_dim
        assert self.review_timestamps < self.history_timestamps, \
            f"Review timestamps {self.review_timestamps} must be less than history timestamps {self.history_timestamps}."

        self.proj = nn.Linear(self.encoder_config['output_dim'],
                              self.output_dim)

    @apply_forward_hook
    def encode(self, features: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Encode the input features and times into a latent representation.
        """
        x = TensorDict({
            'dense': features,
            'sparse': times
        }, batch_size=features.shape[0], device=features.device)

        x = self.encoder_embed(x)

        level, growths, seasons = self.encoder(x, x)

        return level, growths, seasons

    @apply_forward_hook
    def decode(self,
               level: torch.Tensor,
               growths: List[torch.Tensor],
               seasons: List[torch.Tensor],
               ) -> torch.Tensor:
        """
        Decode the input features and times into a seasonal and trend part.
        """

        growth, season = self.decoder(growths, seasons)

        decode_output = level[:, -1:] + growth + season  # Combine level, growth, and season

        return decode_output

    def forward_forecasting(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for forecasting task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        level, growths, seasons = self.encode(x_features, x_times)  # Encode input features and times

        decode_output = self.decode(level, growths, seasons)  # Decode to get seasonal and trend components

        decode_output = self.proj(decode_output)  # Project to output dimension

        return decode_output

    def forward_imputation(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for imputation task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        # Encode input
        level, growths, seasons = self.encode(x_features, x_times)  # Encode input features and times

        decode_output = self.decode(level, growths, seasons)  # Decode to get seasonal and trend components

        decode_output = self.proj(decode_output)  # Project to output dimension

        x = decode_output

        return x

    def forward_outlier(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for anomaly detection task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        # Encode input
        level, growths, seasons = self.encode(x_features, x_times)  # Encode input features and times

        decode_output = self.decode(level, growths, seasons)  # Decode to get seasonal and trend components

        decode_output = self.proj(decode_output)  # Project to output dimension

        x = decode_output

        return x

    def forward_classification(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for classification task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        # Encode input
        level, growths, seasons = self.encode(x_features, x_times)  # Encode input features and times

        decode_output = self.decode(level, growths, seasons)  # Decode to get seasonal and trend components

        decode_output = self.proj(decode_output)  # Project to output dimension

        decode_output = decode_output.mean(dim=1)  # Global average pooling across time steps

        x = decode_output

        return x

    def forward(self, x: TensorDict, target: TensorDict, task: str = None) -> torch.Tensor:
        """
        Forward pass for the Autoformer model.
        """
        if task is not None:
            task_type = TaskType.from_string(task.lower())
        else:
            task_type = self.task_type

        if task_type == TaskType.FORECASTING:
            return self.forward_forecasting(x, target)
        elif task_type == TaskType.IMPUTATION:
            return self.forward_imputation(x, target)
        elif task_type == TaskType.OUTLIER:
            return self.forward_outlier(x, target)
        elif task_type == TaskType.CLASSIFICATION:
            return self.forward_classification(x, target)
        else:
            raise ValueError(f"Unknown task type: {task_type.value}. Supported tasks are: "
                             f"{', '.join([t.value for t in TaskType])}.")


if __name__ == '__main__':
    device = torch.device('cpu')

    batch_size = 4
    embedding_dim = 128
    num_stocks = 6
    history_timestamps = 64
    future_timestamps = 32
    review_timestamps = 32
    input_dim = 145
    dense_input_dim = input_dim
    sparse_input_dim = 4  # e.g., year, month, weekday, day
    input_channel = 1
    latent_dim = embedding_dim
    output_dim = embedding_dim
    predict_dim = num_stocks
    num_heads = 4
    input_length = history_timestamps
    output_length = 32
    k = 10
    dropout = 0.1

    encoder_embed_config = dict(
        type="AggDataEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        dropout=0.1
    )

    encoder_config = dict(
        type='EtsformerEncoder',
        input_dim=embedding_dim,
        latent_dim=embedding_dim,
        output_dim=embedding_dim,
        num_heads=num_heads,
        depth=2,
        output_length=output_length,
        k=k,
        dropout=dropout
    )

    decoder_config = dict(
        type='EtsformerDecoder',
        num_heads=num_heads,
        output_length=output_length,
        depth=2,
        dropout=dropout,
    )

    model = Etsformer(
        task_type='forecasting',
        encoder_embed_config=encoder_embed_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        review_timestamps=review_timestamps,
        output_dim=predict_dim
    ).to(device)

    batch_size = 4
    x_features = torch.randn(batch_size, history_timestamps, num_stocks, dense_input_dim)  # Batch size of 4, 10 time steps, 64 features
    years = torch.randint(2015, 2026, (batch_size, history_timestamps, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, history_timestamps, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, history_timestamps, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, history_timestamps, 1))  # Day feature
    x_times = torch.cat([days, months, weekdays, years], dim=-1).to(
        device)  # Shape: (batch_size, seq_len, num_features)
    y_features = torch.randn(batch_size, future_timestamps, num_stocks, dense_input_dim)  # Target features
    years = torch.randint(2015, 2026, (batch_size, future_timestamps, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, future_timestamps, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, future_timestamps, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, future_timestamps, 1))  # Day feature
    y_times = torch.cat([days, months, weekdays, years], dim=-1).to(device)
    y_times = torch.cat([x_times[:, -review_timestamps:, :], y_times], dim=1)  # Concatenate review timestamps

    x = TensorDict({
        'features': x_features.to(device),
        'times': x_times.to(device)
    }, batch_size=(batch_size,)).to(device)
    target = TensorDict({
        'features': y_features.to(device),
        'times': y_times.to(device)
    }, batch_size=(batch_size,)).to(device)

    output = model(x, target, task='forecasting')
    print(output)
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='imputation')
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='outlier')
    print(f"Output shape: {output.shape}")  # Should be [B, history_timestamps, output_dim]

    output = model(x, target, task='classification')
    print(f"Output shape: {output.shape}")  # Should be [B, num_classes] (output_dim is num_classes in this case)
