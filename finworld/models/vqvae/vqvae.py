from typing import Dict
import torch
import torch.nn as nn
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import MODEL
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import QUANTIZER
from finworld.registry import EMBED

@MODEL.register_module(force=True)
class VQVAE(nn.Module):
    def __init__(self,
                 embed_config: Dict,
                 encoder_config: Dict,
                 quantizer_config: Dict,
                 decoder_config: Dict,
                 ):
        super(VQVAE, self).__init__()

        self.embed_layer = EMBED.build(embed_config)

        self.encoder = ENCODER.build(encoder_config)

        self.quantizer = QUANTIZER.build(quantizer_config)

        self.decoder = DECODER.build(decoder_config)

        self.if_mask =self.encoder.if_mask
        self.patch_size = self.embed_layer.patch_size
        self.output_dim = self.decoder.output_dim

    @apply_forward_hook
    def encode(self, sample: torch.FloatTensor):
        output, mask, ids_restore = self.encoder(sample)

        quantized, embed_ind, quantized_loss, quantized_loss_breakdown = self.quantizer(output)

        weighted_quantized_loss = quantized_loss[0]
        weighted_commit_loss = quantized_loss_breakdown.weighted_commit_loss
        weighted_codebook_diversity_loss = quantized_loss_breakdown.weighted_codebook_diversity_loss
        weighted_orthogonal_reg_loss = quantized_loss_breakdown.weighted_orthogonal_reg_loss

        return_info = dict(
            quantized=quantized,
            embed_ind=embed_ind,
            mask=mask,
            ids_restore=ids_restore,
            weighted_quantized_loss=weighted_quantized_loss,
            weighted_commit_loss=weighted_commit_loss,
            weighted_codebook_diversity_loss=weighted_codebook_diversity_loss,
            weighted_orthogonal_reg_loss=weighted_orthogonal_reg_loss
        )

        return return_info

    @apply_forward_hook
    def decode(self, sample: torch.FloatTensor, ids_restore: torch.LongTensor):

        recon_sample = self.decoder(sample, ids_restore = ids_restore)

        return_info = dict(
            recon_sample=recon_sample
        )

        return return_info

    def forward(self,
                sample: torch.FloatTensor,
                label: torch.LongTensor = None,
                training: bool = True,
                ):

        embed = self.embed_layer(sample)

        encoder_output = self.encode(embed)

        quantized = encoder_output["quantized"]
        mask = encoder_output["mask"]
        ids_restore = encoder_output["ids_restore"]
        weighted_quantized_loss = encoder_output["weighted_quantized_loss"]
        weighted_commit_loss = encoder_output["weighted_commit_loss"]
        weighted_codebook_diversity_loss = encoder_output["weighted_codebook_diversity_loss"]
        weighted_orthogonal_reg_loss = encoder_output["weighted_orthogonal_reg_loss"]

        decoder_output = self.decode(quantized, ids_restore=ids_restore)
        recon_sample = decoder_output["recon_sample"]

        return_info = dict(
            recon_sample=recon_sample,
            mask = mask,
            ids_restore = ids_restore,
            weighted_quantized_loss = weighted_quantized_loss,
            weighted_commit_loss = weighted_commit_loss,
            weighted_codebook_diversity_loss = weighted_codebook_diversity_loss,
            weighted_orthogonal_reg_loss = weighted_orthogonal_reg_loss
        )

        return return_info


if __name__ == '__main__':
    device = torch.device("cpu")

    embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(4, 1, 152),
        input_channel=1,
        input_dim=152,
        output_dim=128,
        temporal_dim=3,
    )

    encoder_config = dict(
        type='VAETransformerEncoder',
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=128,
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
    )

    quantizer_config = dict(
        type="VectorQuantizer",
        dim=128,
        codebook_size=512,
        codebook_dim=128,
        decay=0.99,
        commitment_weight=1.0
    )

    decoder_config = dict(
        type='VAETransformerDecoder',
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        output_dim=5,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        cls_embed=True,
        sep_pos_embed=True,
        trunc_init=False,
        no_qkv_bias=False,
    )

    vae = VQVAE(embed_config,
                encoder_config,
                quantizer_config,
                decoder_config).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    sample = batch.to(torch.float32)

    output = vae(batch)
    print(output["recon_sample"].shape)