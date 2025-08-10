import torch
from torch import nn
from typing import List
from timm.models.layers import to_2tuple

from finworld.registry import EMBED, MODEL
from finworld.utils import modulate
from finworld.models import DiTBlock as Block


@MODEL.register_module(force=True)
class DiT(nn.Module):
    def __init__(self,
                 *args,
                 embed_config: dict = None,
                 timestep_embed_config: dict = None,
                 label_embed_config: dict = None,
                 text_encoder_config: dict = None,
                 if_label_embed: bool = True,
                 if_text_encoder: bool = True,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 output_dim: int = 256,
                 depth: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 norm_layer = nn.LayerNorm,
                 cls_embed: bool = True,
                 sep_pos_embed: bool = True,
                 trunc_init: bool = False,
                 no_qkv_bias: bool = False,
                 **kwargs
                 ):
        super(DiT, self).__init__()

        self.if_label_embed = if_label_embed
        self.if_text_encoder = if_text_encoder

        self.data_size = to_2tuple(embed_config.get('data_size', None))
        self.patch_size = to_2tuple(embed_config.get('patch_size', None))

        self.input_size = (self.data_size[0] // self.patch_size[0], self.data_size[1] // self.patch_size[1])
        self.num_patches = self.input_size[0] * self.input_size[1]

        self.to_latent = nn.Linear(input_dim, latent_dim)
        self.timestep_embed_layer = EMBED.build(timestep_embed_config)
        self.timestep_to_latent = nn.Linear(self.timestep_embed_layer.embed_dim, latent_dim)
        if self.if_label_embed:
            self.label_embed_layer = EMBED.build(label_embed_config)
            self.label_to_latent = nn.Linear(self.label_embed_layer.embed_dim, latent_dim)
        if self.if_text_encoder:
            self.text_encoder = MODEL.build(text_encoder_config)
            self.text_to_latent = nn.Linear(self.text_encoder.embed_dim, latent_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else latent_dim * 2
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.input_size[1], latent_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.input_size[0], latent_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, latent_dim))
        else:
            if self.cls_embed:
                _num_patches = self.num_patches + 1
            else:
                _num_patches = self.num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, latent_dim),
            )

        self.blocks = nn.ModuleList(
            [
                Block(latent_dim,
                      num_heads,
                      mlp_ratio,
                      qkv_bias=not no_qkv_bias,
                      norm_layer=norm_layer)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(latent_dim)

        self.ada_layer_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim, bias=True)
        )

        self.pred = nn.Linear(
            latent_dim,
            output_dim,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,
                sample: torch.FloatTensor,
                timestep: torch.LongTensor = None,
                label: torch.LongTensor = None,
                text: List[str] = None):
        """
        :param x: (N, L, D) tensor of spatial inputs (images or latent representations of the inputs)
        :param timestep: (N, ) tensor of diffusion timesteps
        :param label: (N, ) tensor of class labels
        :param text: list of N strings of text prompts
        :return:
        """
        timestep = self.timestep_embed_layer(timestep).to(sample.device, sample.dtype)
        timestep = self.timestep_to_latent(timestep)
        if self.if_label_embed:
            label = self.label_embed_layer(label, self.training).to(sample.device, sample.dtype)
            label = self.label_to_latent(label)
        if self.if_text_encoder:
            text = self.text_encoder(text).to(sample.device, sample.dtype)
            text = self.text_to_latent(text)

        sample = self.to_latent(sample)

        N, L, D = sample.shape

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(sample.shape[0], -1, -1)
            sample = torch.cat((cls_tokens, sample), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1],
                dim=1,
            )

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(
                            pos_embed.shape[0], -1, -1
                        ),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]

        # add pos embed
        sample = sample + pos_embed

        attn = self.blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            sample = sample.view([N, L, D])

        condition = timestep
        if self.if_label_embed:
            condition = condition + label
        if self.if_text_encoder:
            condition = condition + text

        # apply Transformer blocks
        for blk in self.blocks:
            sample = blk(sample, condition)

        shift, scale = self.ada_layer_norm(condition).chunk(2, dim= 1)

        sample  = self.norm(sample)

        sample = modulate(sample, shift, scale)

        # predictor projection
        sample = self.pred(sample)

        if self.cls_embed:
            # remove cls token
            sample = sample[:, 1:, :]
        else:
            sample = sample[:, :, :]

        return sample

    def forward_with_scale(self, x, t, y, ids_restore=None, scale = 1):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, ids_restore)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


if __name__ == '__main__':
    from finworld.models import VAETransformerEncoder
    from finworld.models import DiagonalGaussianDistribution
    from finworld.models import VAETransformerEncoderOutput

    device = torch.device("cpu")

    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 153),
        patch_size=(4, 153),
        input_channel=1,
        input_dim=153,
        output_dim=128,
        temporal_dim=3,
    )
    encoder = VAETransformerEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=128 * 2,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    ).to(device)

    feature = torch.randn(4, 1, 64, 150)
    temporal = torch.zeros(4, 1, 64, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)

    output, mask, ids_restore = encoder(batch)
    print(output.shape)
    print(mask.shape)
    print(ids_restore.shape)

    moments = output

    posterior = DiagonalGaussianDistribution(moments)
    extra_info = dict(mask=mask, ids_restore=ids_restore)
    output = VAETransformerEncoderOutput(latent_dist=posterior,
                                         latent_embeds=moments,
                                         extra_info=extra_info)
    sample = posterior.sample()
    print(sample.shape)

    timestep_embed_config = dict(
        type="TimestepEmbed",
        embed_dim=128,
        frequency_embedding_size=256
    )
    label_embed_config = dict(
        type="LabelEmbed",
        embed_dim=128,
        num_classes=30,
        dropout_prob=0.1
    )
    text_encoder_config = dict(
        type="OpenAITextEncoder",
        provider_cfg_path="configs/openai_config.json",
        if_reduce_dim=True,
        reduced_dim=128,
    )
    dit = DiT(
        embed_config=embed_config,
        timestep_embed_config=timestep_embed_config,
        label_embed_config=label_embed_config,
        text_encoder_config=text_encoder_config,
        if_label_embed=True,
        if_text_encoder=True,
        input_dim=128,
        latent_dim=128,
        output_dim=256,
        depth = 2,
        num_heads = 4,
        mlp_ratio = 4.0,
        cls_embed = True,
        sep_pos_embed = True,
        trunc_init = False,
        no_qkv_bias = False,
    ).to(device)

    timestep = torch.zeros((4, )).long().to(device)
    label = torch.zeros((4, )).long().to(device)
    text = ["sample"] * 4

    output = dit(sample, timestep, label, text)
    print(output.shape)