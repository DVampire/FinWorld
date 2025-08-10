import torch
from torch import nn
from typing import Dict
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
class Crossformer(nn.Module):
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
        super(Crossformer, self).__init__(*args, **kwargs)

        self.task_type = TaskType.from_string(task_type)

        self.encoder_embed_config = encoder_embed_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.encode_seg_num = self.encoder_config["seg_num"]
        self.decode_seg_num = self.decoder_config["seg_num"]

        self.encoder_embed = EMBED.build(encoder_embed_config)
        self.encoder = ENCODER.build(encoder_config)

        self.data_size = self.encoder_embed.data_size
        self.patch_size = self.encoder_embed.patch_size

        self.encoder_position_embed = AbsPosition2DEmbed(
            num_time = history_timestamps,
            num_space = self.data_size[1],  # Number of spatial features (e.g., stocks)
            embed_dim = self.encoder_embed.output_dim,  # Embedding dimension
        )

        self.decoder = DECODER.build(decoder_config)

        self.decoder_position_embed = AbsPosition2DEmbed(
            num_time = future_timestamps,
            num_space = self.data_size[1],  # Number of spatial features (e.g., stocks)
            embed_dim = self.decoder.output_dim,  # Embedding dimension
        )

        self.history_timestamps = history_timestamps
        self.future_timestamps = future_timestamps
        self.review_timestamps = review_timestamps  # Review timestamps from history
        self.output_dim = output_dim
        assert self.review_timestamps < self.history_timestamps, \
            f"Review timestamps {self.review_timestamps} must be less than history timestamps {self.history_timestamps}."

        self.proj = nn.Linear(
            self.decoder.output_dim,
            self.patch_size[0],  # Project to output dimension for each stock
            bias=False
        )

    @apply_forward_hook
    def encode(self, features: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Encode the input features and times into a latent representation.
        """
        x = self.encoder_embed(features)

        B, T, N, C = x.shape  # Batch size, patched time steps, number of stocks, embedding dimension
        x = rearrange(x, 'b t n c -> b (t n) c')  # Reshape to (batch_size, time_steps * num_stocks, embedding_dim)
        x = self.encoder_position_embed(x)
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N)  # Reshape back to (batch_size, time_steps, num_stocks, embedding_dim)

        enc_outs = self.encoder(x)

        return enc_outs

    @apply_forward_hook
    def decode(self,
               features: torch.Tensor,
               times: torch.Tensor,
               cross: torch.Tensor,
               ) -> torch.Tensor:
        """
        Decode the input features and times into a seasonal and trend part.
        """
        B, T, N, C = features.shape  # Batch size, future time steps, number of stocks, embedding dimension
        x = rearrange(features, 'b t n c -> b (t n) c')
        x = self.decoder_position_embed(x)
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N)  # Reshape back to (batch_size, future_time_steps, num_stocks, embedding_dim)

        x = self.decoder(x, cross)  # Decode using the decoder

        x = self.proj(x)  # Project to output dimension

        x = rearrange(x, 'b t n c -> b (t c) n', c = self.patch_size[0])

        return x

    def forward_forecasting(self, x: TensorDict, target: TensorDict) -> torch.Tensor:
        """
        Forward pass for forecasting task.
        """
        x_features = x['features']  # input features
        x_times = x['times']  # input time features
        target_features = target['features']  # target features
        target_times = target['times']  # target time features

        encode_outputs = self.encode(x_features, x_times)  # Encode input features and times

        cross = encode_outputs
        decode_in = torch.zeros((
            x_features.shape[0],
            self.decode_seg_num,
            self.data_size[1],
            self.encoder_embed.output_dim
        )).to(x_features.device)

        decode_output = self.decode(decode_in, target_times, cross)  # Decode

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
        encode_outputs = self.encode(x_features, x_times)
        encode_output = encode_outputs[-1]

        decode_output = self.proj(encode_output)  # Project to output dimension
        decode_output = rearrange(decode_output, 'b t n c -> b (t c) n', c=self.patch_size[0])

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
        encode_outputs = self.encode(x_features, x_times)
        encode_output = encode_outputs[-1]

        decode_output = self.proj(encode_output)  # Project to output dimension
        decode_output = rearrange(decode_output, 'b t n c -> b (t c) n', c=self.patch_size[0])

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
        encode_outputs = self.encode(x_features, x_times)
        encode_output = encode_outputs[-1]

        decode_output = self.proj(encode_output)  # Project to output dimension
        decode_output = rearrange(decode_output, 'b t n c -> b (t c) n', c=self.patch_size[0])

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
    num_stocks = 6
    history_timestamps = 64
    future_timestamps = 32
    review_timestamps = 32
    data_size = (history_timestamps, num_stocks, 145)
    patch_size = (4, 1, 145)
    predict_dim = num_stocks
    input_dim = 145
    dense_input_dim = input_dim
    input_channel = 1
    latent_dim = 128
    output_dim = 128
    encode_seg_num = history_timestamps // patch_size[0]  # Number of segments based on patch size
    decode_seg_num = future_timestamps // patch_size[0]  # Number of segments for decoding
    window_size = 1
    factor = 1

    encoder_embed_config = dict(
        type='TimePatchEmbed',
        data_size=data_size,
        patch_size=patch_size,
        input_dim=145,
        input_channel=1,
        embed_dim=latent_dim,
        if_use_stem=True,
        stem_embedding_dim=64
    )

    encoder_config = dict(
        type='CrossformerEncoder',
        input_dim=latent_dim,
        latent_dim=latent_dim,
        output_dim=latent_dim,
        seg_num=encode_seg_num,
        window_size=window_size,
        depth=2,
        factor=factor
    )

    decoder_config = dict(
        type='CrossformerDecoder',
        input_dim=latent_dim,
        latent_dim=latent_dim,
        output_dim=latent_dim,
        seg_num=decode_seg_num,
        window_size=window_size,
        depth=2,
        factor=factor
    )

    model = Crossformer(
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
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='imputation')
    print(f"Output shape: {output.shape}")  # Should be [B, future_timestamps, output_dim]

    output = model(x, target, task='outlier')
    print(f"Output shape: {output.shape}")  # Should be [B, history_timestamps, output_dim]

    output = model(x, target, task='classification')
    print(f"Output shape: {output.shape}")  # Should be [B, num_classes] (output_dim is num_classes in this case)
