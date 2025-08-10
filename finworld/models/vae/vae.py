from typing import Dict

import torch
import torch.nn as nn
import numpy as np
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import MODEL
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import EMBED
from finworld.models import DiagonalGaussianDistribution
from finworld.models.embed import unpatchify

@MODEL.register_module(force=True)
class TransformerVAE(nn.Module):
    def __init__(self,
                 *args,
                 embed_config: Dict,
                 encoder_config: Dict,
                 decoder_config: Dict,
                 output_dim: int = 4,
                 **kwargs
                 ):
        super(TransformerVAE, self).__init__(*args, **kwargs)

        self.embed = EMBED.build(embed_config)

        self.patch_info = self.embed.patch_info
        self.data_size = self.embed.data_size
        self.patch_size = self.embed.patch_size
        self.input_channel = self.embed.input_channel

        self.encoder = ENCODER.build(encoder_config)
        self.decoder = DECODER.build(decoder_config)

        self.output_dim = output_dim

        self.rec = nn.Linear(
            self.decoder.output_dim,
            int(np.prod([self.input_channel] + list(self.patch_size))),
            bias=True,
        )

        self.proj = nn.Linear(self.data_size[-1], self.output_dim)

    @apply_forward_hook
    def encode(self, sample: torch.FloatTensor):

        output, mask, ids_restore = self.encoder(sample)

        moments = output
        posterior = DiagonalGaussianDistribution(moments)

        return_info = dict(output = output,
                           posterior=posterior,
                           mask=mask,
                           ids_restore=ids_restore)

        return return_info

    @apply_forward_hook
    def decode(self, sample: torch.FloatTensor, ids_restore: torch.LongTensor):
        output = self.decoder(sample, ids_restore = ids_restore)
        return_info = dict(output=output)
        return return_info

    def forward(self, sample: torch.FloatTensor):

        sample = self.embed(sample)

        enc_res_info = self.encode(sample)

        posterior = enc_res_info["posterior"]
        mask = enc_res_info["mask"]
        ids_restore = enc_res_info["ids_restore"]

        sample_ = posterior.sample()

        dec_res_info = self.decode(sample_, ids_restore=ids_restore)
        dec_output = dec_res_info["output"]

        recon_output = self.rec(dec_output)
        recon_sample = unpatchify(recon_output, patch_info=self.patch_info)
        recon_sample = self.proj(recon_sample)

        return_info = dict(
            recon_sample = recon_sample,
            posterior = posterior,
            mask=mask,
            ids_restore=ids_restore
        )

        return return_info


if __name__ == '__main__':
    device = torch.device("cpu")

    # For 3D
    input_channel = 1
    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel=input_channel,
        input_dim=152,
        output_dim=128,
        temporal_dim=3,
    )
    encoder_config = dict(
        type='TransformerEncoder',
        input_dim=128,
        latent_dim=128,
        output_dim=128 * 2,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    )

    decoder_config = dict(
        type='TransformerDecoder',
        input_dim=128,
        latent_dim=128,
        output_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
    )

    vae = VAE(embed_config,
              encoder_config,
              decoder_config,
              output_dim=4).to(device)
    feature = torch.randn(4, 1, 64, 29, 152).to(device)
    output = vae(feature)
    print(output["recon_sample"].shape)

    # For 2D
    input_channel = 1
    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 152),
        patch_size=(4, 152),
        input_channel=input_channel,
        input_dim=152,
        output_dim=128
    )
    encoder_config = dict(
        type='TransformerEncoder',
        input_dim=128,
        latent_dim=128,
        output_dim=128 * 2,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
        if_mask=False,
        mask_ratio_min=0.5,
        mask_ratio_max=1.0,
        mask_ratio_mu=0.55,
        mask_ratio_std=0.25,
    )

    decoder_config = dict(
        type='TransformerDecoder',
        input_dim=128,
        latent_dim=128,
        output_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        if_remove_cls_embed=True,
        no_qkv_bias=False,
        trunc_init=False,
    )

    vae = TransformerVAE(
        embed_config=embed_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        output_dim=4
    ).to(device)

    feature = torch.randn(4, 1, 64, 152).to(device)
    output = vae(feature)
    print(output["recon_sample"].shape)