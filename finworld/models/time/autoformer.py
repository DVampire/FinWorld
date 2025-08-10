import torch
from torch import nn
from typing import Dict
from tensordict import TensorDict
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import EMBED
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import MODEL
from finworld.models.modules.autoformer import SeriesDecomp
from finworld.task import TaskType

@MODEL.register_module(force=True)
class Autoformer(nn.Module):
    def __init__(self,
                 *args,
                 task_type: str = 'forecasting',
                 encoder_embed_config: Dict,
                 decoder_embed_config: Dict,
                 encoder_config: Dict,
                 decoder_config: Dict,
                 history_timestamps: int = 64,
                 future_timestamps: int = 32,
                 review_timestamps: int = 10,
                 moving_avg: int = 25,
                 output_dim: int = 4,
                 **kwargs
                 ) -> None:
        super(Autoformer, self).__init__(*args, **kwargs)

        self.task_type = TaskType.from_string(task_type)

        self.encoder_embed_config = encoder_embed_config
        self.decoder_embed_config = decoder_embed_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.encoder_embed = EMBED.build(encoder_embed_config)
        self.decoder_embed = EMBED.build(decoder_embed_config)
        self.encoder = ENCODER.build(encoder_config)
        self.decoder = DECODER.build(decoder_config)

        self.decomp = SeriesDecomp(
            kernel_size=moving_avg
        )

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.review_timestamps = review_timestamps # Review timestamps from history
        assert self.review_timestamps < self.history_timestamps, \
            f"Review timestamps {self.review_timestamps} must be less than history timestamps {self.history_timestamps}."

        self.proj = nn.Linear(
            self.decoder.output_dim,
            output_dim,
            bias=False
        )

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

        x = self.encoder(x, attn_mask=None)

        return x

    @apply_forward_hook
    def decode(self,
               features: torch.Tensor,
               times: torch.Tensor,
               cross: torch.Tensor,
               trend: torch.Tensor
               ) -> torch.Tensor:
        """
        Decode the input features and times into a seasonal and trend part.
        """
        x = TensorDict({
            'dense': features,
            'sparse': times
        }, batch_size=features.shape[0], device=features.device)

        x = self.decoder_embed(x)

        seasonal_part, trend_part = self.decoder(x,
                                                 cross,
                                                 attn_mask=None,
                                                 cross_attn_mask=None,
                                                 )

        x = trend_part + seasonal_part
        return x

    def forward_forecasting(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for forecasting task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        # Initialize mean and zeros for seasonal and trend components
        mean = torch.mean(x_features, dim=1, keepdim=True).repeat_interleave(self.future_timestamps, dim=1)

        size = (*target_features.shape[:1], self.future_timestamps, *target_features.shape[2:])
        zeros = torch.zeros(size, device=x_features.device)

        # Decode input
        seasonal_init, trend_init = self.decomp(x_features)

        trend_init = torch.cat([trend_init[:, -self.review_timestamps:, ...], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.review_timestamps:, ...], zeros], dim=1)

        # Encode input
        encode_output = self.encode(x_features, x_times)

        # Decode target
        decode_output = self.decode(features=seasonal_init,
                                    times=target_times,
                                    cross=encode_output,
                                    trend=trend_init)

        decode_output = self.proj(decode_output)  # Project to output dimension

        # Extract the decoded output for the future timestamps
        x = decode_output[:, -self.future_timestamps:, ...]
        return x

    def forward_imputation(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for imputation task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        # Encode input
        encode_output = self.encode(x_features, x_times)

        decode_output = self.proj(encode_output)  # Project to output dimension

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
        encode_output = self.encode(x_features, x_times)

        decode_output = self.proj(encode_output)  # Project to output dimension

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
        encode_output = self.encode(x_features, x_times)

        decode_output = self.proj(encode_output)  # (batch_size, seq_length, output_dim)

        decode_output = decode_output.mean(dim=1)  # Global average pooling over time dimension

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

def test_agg_autuformer():
    device = torch.device("cpu")

    history_timestamps = 64
    future_timestamps = 32
    review_timestamps = 32
    dense_input_dim = 145
    sparse_input_dim = 4
    num_asset = 6
    latent_dim = 128
    output_dim = 128
    predict_dim = num_asset

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

    decoder_embed_config = dict(
        type="AggDataEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        dropout=0.1
    )

    encoder_config = dict(
        type="AutoformerEncoder",
        input_dim=output_dim,
        latent_dim=output_dim,
        output_dim=output_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        no_qkv_bias=False,
        moving_avg=25,
        factor=1
    )

    decoder_config = dict(
        type="AutoformerDecoder",
        input_dim=output_dim,
        latent_dim=output_dim,
        output_dim=output_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        no_qkv_bias=False,
        moving_avg=25,
        factor=1
    )

    model = Autoformer(
        task_type='forecasting',
        encoder_embed_config=encoder_embed_config,
        decoder_embed_config=decoder_embed_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        review_timestamps=review_timestamps,
        moving_avg=25,
        output_dim=predict_dim
    )

    batch_size = 4

    x_features = torch.randn(batch_size, history_timestamps, num_asset,
                             dense_input_dim)  # Batch size of 4, 10 time steps, 64 features
    years = torch.randint(2015, 2026, (batch_size, history_timestamps, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, history_timestamps, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, history_timestamps, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, history_timestamps, 1))  # Day feature
    x_times = torch.cat([days, months, weekdays, years], dim=-1).to(
        device)  # Shape: (batch_size, seq_len, num_features)

    y_features = torch.randn(batch_size, future_timestamps, num_asset, dense_input_dim)  # Target features
    years = torch.randint(2015, 2026, (batch_size, future_timestamps, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, future_timestamps, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, future_timestamps, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, future_timestamps, 1))  # Day feature
    y_times = torch.cat([days, months, weekdays, years], dim=-1).to(device)
    y_times = torch.cat([x_times[:, -review_timestamps:, ...], y_times], dim=1)  # Concatenate review timestamps

    x = TensorDict({
        'features': x_features.to(device),
        'times': x_times.to(device)
    }, batch_size=(batch_size,)).to(device)
    target = TensorDict({
        'features': y_features.to(device),
        'times': y_times.to(device)
    }, batch_size=(batch_size,)).to(device)

    output = model(x, target, task='forecasting')
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='imputation')
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='outlier')
    print(f"Output shape: {output.shape}")  # Should be [B, history_timestamps, output_dim]

    output = model(x, target, task='classification')
    print(f"Output shape: {output.shape}")  # Should be [B, num_classes] (output_dim is num_classes in this case)

def test_autoformer():
    device = torch.device("cpu")

    history_timestamps = 64
    future_timestamps = 32
    review_timestamps = 32
    dense_input_dim = 145
    sparse_input_dim = 4
    output_dim = 128
    predict_dim = 6

    encoder_embed_config = dict(
        type="DataEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        dropout=0.1
    )

    decoder_embed_config = dict(
        type="DataEmbed",
        dense_input_dim=dense_input_dim,
        sparse_input_dim=sparse_input_dim,
        output_dim=output_dim,
        start_timestamp="2015-05-01",
        end_timestamp="2025-05-01",
        level='1day',
        dropout=0.1
    )

    encoder_config = dict(
        type="AutoformerEncoder",
        input_dim=output_dim,
        latent_dim=output_dim,
        output_dim=output_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        no_qkv_bias=False,
        moving_avg=25,
        factor=1
    )

    decoder_config = dict(
        type="AutoformerDecoder",
        input_dim=output_dim,
        latent_dim=output_dim,
        output_dim=output_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        no_qkv_bias=False,
        moving_avg=25,
        factor=1
    )

    model = Autoformer(
        task_type='forecasting',
        encoder_embed_config=encoder_embed_config,
        decoder_embed_config=decoder_embed_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        review_timestamps=review_timestamps,
        moving_avg=25,
        output_dim=predict_dim
    )

    batch_size = 4
    x_features = torch.randn(batch_size, history_timestamps,
                             dense_input_dim)  # Batch size of 4, 10 time steps, 64 features
    years = torch.randint(2015, 2026, (batch_size, history_timestamps, 1))  # Year feature
    months = torch.randint(1, 13, (batch_size, history_timestamps, 1))  # Month feature
    weekdays = torch.randint(0, 7, (batch_size, history_timestamps, 1))  # Weekday feature
    days = torch.randint(1, 32, (batch_size, history_timestamps, 1))  # Day feature
    x_times = torch.cat([days, months, weekdays, years], dim=-1).to(
        device)  # Shape: (batch_size, seq_len, num_features)
    y_features = torch.randn(batch_size, future_timestamps, dense_input_dim)  # Target features
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
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='imputation')
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='outlier')
    print(f"Output shape: {output.shape}")  # Should be [B, history_timestamps, output_dim]

    output = model(x, target, task='classification')
    print(f"Output shape: {output.shape}")  # Should be [B, num_classes] (output_dim is num_classes in this case)
    

if __name__ == '__main__':

    test_agg_autuformer()
    test_autoformer()