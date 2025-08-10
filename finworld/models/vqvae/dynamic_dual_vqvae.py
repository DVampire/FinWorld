import torch
import torch.nn as nn
from typing import Dict
from torch.nn import functional as F
from einops import rearrange
from diffusers.utils.accelerate_utils import apply_forward_hook
from vit_pytorch.cross_vit import MultiScaleEncoder

from finworld.registry import MODEL
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import QUANTIZER
from finworld.registry import EMBED
from finworld.models.modules.clip import CLIPLayer
from finworld.models.modules.transformer import Mlp
from finworld.models.modules.distribution import DiagonalGaussianDistribution

@MODEL.register_module(force=True)
class DynamicDualVQVAE(nn.Module):
    def __init__(self,
                 cs_embed_config: Dict = None, # cross-sectional embedding config
                 ts_embed_config: Dict = None, # time-series embedding config
                 cs_config: Dict = None, # cross-sectional config
                 ts_config: Dict = None, # time-series config
                 multi_scale_encoder_config: Dict = None, # multi-scale encoder config
                 if_use_multi_scale_encoder: bool = True, # if use multi-scale encoder
                 asset_num: int = 29, # asset number
                 cl_loss_weight: float = 1.0, # clip loss weight
                 temperature: float = 1.0, # temperature
                 output_dim: int = 4, # output dimension
                 ):
        super(DynamicDualVQVAE, self).__init__()

        self.asset_num = asset_num
        self.output_dim = output_dim

        self.cs_encoder_config = cs_config.get("cs_encoder_config", {}) # cross-sectional factor encoder config
        self.cs_quantizer_config = cs_config.get("cs_quantizer_config", {}) # cross-sectional factor quantizer config
        self.cs_decoder_config = cs_config.get("cs_decoder_config", {}) # cross-sectional reconstruction decoder config

        self.ts_encoder_config = ts_config.get("ts_encoder_config", {}) # time-series factor encoder config
        self.ts_quantizer_config = ts_config.get("ts_quantizer_config", {}) # time-series factor quantizer config
        self.ts_decoder_config = ts_config.get("ts_decoder_config", {}) # time-series reconstruction decoder config

        self.cs_embed_layer = EMBED.build(cs_embed_config)
        self.cs_encoder = ENCODER.build(self.cs_encoder_config)
        self.cs_quantizer = QUANTIZER.build(self.cs_quantizer_config)
        self.cs_decoder = DECODER.build(self.cs_decoder_config)
        self.cs_if_mask = self.cs_encoder.if_mask
        self.cs_patch_size = self.cs_embed_layer.patch_size
        self.cs_output_dim = self.cs_decoder.output_dim
        self.cs_num_patches = self.cs_embed_layer.patch_info["num_patches"]
        cs_output_dim = self.cs_embed_layer.patch_size[0] * self.cs_embed_layer.patch_size[1] * self.output_dim
        self.cs_proj = nn.Linear(self.cs_decoder.output_dim, cs_output_dim)

        self.ts_embed_layer = EMBED.build(ts_embed_config)
        self.ts_encoder = ENCODER.build(self.ts_encoder_config)
        self.ts_quantizer = QUANTIZER.build(self.ts_quantizer_config)
        self.ts_decoder = DECODER.build(self.ts_decoder_config)
        self.ts_if_mask = self.ts_encoder.if_mask
        self.ts_patch_size = self.ts_embed_layer.patch_size
        self.ts_output_dim = self.ts_decoder.output_dim
        self.ts_num_patches = self.ts_embed_layer.patch_info["num_patches"]
        ts_output_dim = self.ts_embed_layer.patch_size[0] * self.ts_embed_layer.patch_size[1] * self.output_dim
        self.ts_proj = nn.Linear(self.ts_decoder.output_dim, ts_output_dim)

        self.if_use_multi_scale_encoder = if_use_multi_scale_encoder

        if self.if_use_multi_scale_encoder:
            self.multi_scale_encoder = MultiScaleEncoder(**multi_scale_encoder_config)
        else:
            self.multi_scale_encoder = None

        self.clip_layer = CLIPLayer(temperature=temperature, cl_loss_weight=cl_loss_weight)

        latent_dim = self.cs_encoder.latent_dim * 2
        factor_num = self.cs_num_patches + self.ts_num_patches

        # to post distribution encode latent
        self.to_pd_encode_latent = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim,
            out_features=asset_num,
            act_layer=nn.LeakyReLU,
        )
        self.post_distribution_layer = Mlp(
            in_features=factor_num,
            hidden_features=latent_dim,
            out_features=factor_num * 2,
            act_layer=nn.LeakyReLU,
        )

        # to post distribution decode latent
        self.to_pd_decode_latent = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim,
            out_features=asset_num,
            act_layer=nn.LeakyReLU
        )
        self.alpha_distribution_layer = Mlp(
            in_features=factor_num,
            hidden_features=latent_dim,
            out_features=2,
            act_layer=nn.LeakyReLU,
        )
        self.beta_layer = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim,
            out_features=asset_num,
            act_layer=nn.LeakyReLU
        )

        self.multi_head_attention_layer = nn.MultiheadAttention(embed_dim=factor_num,
                                                                num_heads=factor_num)
        self.prior_distribution_layer = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim,
            out_features=2,
            act_layer=nn.LeakyReLU,
        )

    @apply_forward_hook
    def encode(self, sample: torch.FloatTensor):

        cs_embed = self.cs_embed_layer(sample)
        ts_embed = self.ts_embed_layer(sample)

        enc_cs, mask_cs, id_restore_cs = self.cs_encoder(cs_embed)
        enc_ts, mask_ts, id_restore_ts = self.ts_encoder(ts_embed)

        if self.if_use_multi_scale_encoder and self.multi_scale_encoder:
            enc_cs, enc_ts = self.multi_scale_encoder(enc_cs, enc_ts)

        clip_loss = self.clip_layer(enc_cs, enc_ts)
        clip_loss = clip_loss["weighted_cl_loss"]

        quantized_cs, embed_ind_cs, quantized_loss_cs, quantized_loss_breakdown_cs = self.cs_quantizer(enc_cs)
        weighted_quantized_loss_cs = quantized_loss_cs[0]
        weighted_commit_loss_cs = quantized_loss_breakdown_cs.weighted_commit_loss
        weighted_codebook_diversity_loss_cs = quantized_loss_breakdown_cs.weighted_codebook_diversity_loss
        weighted_orthogonal_reg_loss_cs = quantized_loss_breakdown_cs.weighted_orthogonal_reg_loss

        quantized_ts, embed_ind_ts, quantized_loss_ts, quantized_loss_breakdown_ts = self.ts_quantizer(enc_ts)
        weighted_quantized_loss_ts = quantized_loss_ts[0]
        weighted_commit_loss_ts = quantized_loss_breakdown_ts.weighted_commit_loss
        weighted_codebook_diversity_loss_ts = quantized_loss_breakdown_ts.weighted_codebook_diversity_loss
        weighted_orthogonal_reg_loss_ts = quantized_loss_breakdown_ts.weighted_orthogonal_reg_loss

        return_info = dict(
            enc_cs=enc_cs,
            quantized_cs=quantized_cs,
            embed_ind_cs=embed_ind_cs,
            mask_cs=mask_cs,
            id_restore_cs=id_restore_cs,
            enc_ts=enc_ts,
            quantized_ts=quantized_ts,
            embed_ind_ts=embed_ind_ts,
            mask_ts=mask_ts,
            id_restore_ts=id_restore_ts,
            weighted_quantized_loss_cs=weighted_quantized_loss_cs,
            weighted_commit_loss_cs=weighted_commit_loss_cs,
            weighted_codebook_diversity_loss_cs=weighted_codebook_diversity_loss_cs,
            weighted_orthogonal_reg_loss_cs=weighted_orthogonal_reg_loss_cs,
            weighted_quantized_loss_ts=weighted_quantized_loss_ts,
            weighted_commit_loss_ts=weighted_commit_loss_ts,
            weighted_codebook_diversity_loss_ts=weighted_codebook_diversity_loss_ts,
            weighted_orthogonal_reg_loss_ts=weighted_orthogonal_reg_loss_ts,
            clip_loss=clip_loss
        )

        return return_info

    @apply_forward_hook
    def encode_post_distribution(self,
                                 factors: torch.Tensor,
                                 label: torch.FloatTensor = None):

        label = rearrange(label, 'n t s-> (n t) s')
        label = label.unsqueeze(-1)

        portfolio_weights = self.to_pd_encode_latent(factors)
        portfolio_weights = F.softmax(portfolio_weights, dim=-1)
        returns  = torch.matmul(portfolio_weights, label).squeeze(-1)
        moments = self.post_distribution_layer(returns)

        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    @apply_forward_hook
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    @apply_forward_hook
    def decode_post_distribution(self,
                                 factors: torch.Tensor,
                                 mu_post: torch.FloatTensor,
                                 sigma_post: torch.FloatTensor):

        alpha_latent_features = self.to_pd_decode_latent(factors)
        alpha_latent_features = alpha_latent_features.permute(0, 2, 1)
        alpha = self.alpha_distribution_layer(alpha_latent_features)
        alpha = alpha.permute(0, 2, 1)
        moments = rearrange(alpha, 'n c t -> n (c t)')

        alpha_dist = DiagonalGaussianDistribution(moments)

        mu_alpha, sigma_alpha = alpha_dist.mean, alpha_dist.std

        beta = self.beta_layer(factors)
        beta = beta.permute(0, 2, 1)

        y_mu = torch.bmm(beta, mu_post.unsqueeze(-1)) + mu_alpha.unsqueeze(-1)

        sigma_alpha_pow = sigma_alpha.unsqueeze(-1).pow(2)
        beta_pow = beta.pow(2)
        sigma_post_pow = sigma_post.unsqueeze(-1).pow(2)

        sigma_post_pow = torch.bmm(beta_pow, sigma_post_pow)

        y_sigma = torch.sqrt(sigma_alpha_pow + sigma_post_pow)

        sample = self.reparameterize(y_mu, y_sigma)
        sample = sample.squeeze(-1)

        return sample

    @apply_forward_hook
    def encode_prior_distribution(self,
                                  factors: torch.Tensor):

        latent_features = factors.permute(0, 2, 1)
        latent_features = self.multi_head_attention_layer(latent_features, latent_features, latent_features)[0]
        latent_features = latent_features.permute(0, 2, 1)

        latent_features = self.prior_distribution_layer(latent_features)
        latent_features = latent_features.permute(0, 2, 1)

        moments = rearrange(latent_features, 'n c t -> n (c t)')

        prior = DiagonalGaussianDistribution(moments)

        return prior

    @apply_forward_hook
    def decode(self,
               quantized_cs: torch.FloatTensor,
               ids_restore_cs: torch.LongTensor,
               quantized_ts: torch.FloatTensor,
               ids_restore_ts: torch.LongTensor):

        recon_cs = self.cs_decoder(quantized_cs, ids_restore=ids_restore_cs)
        recon_ts = self.ts_decoder(quantized_ts, ids_restore=ids_restore_ts)

        recon_cs = self.cs_proj(recon_cs)
        recon_ts = self.ts_proj(recon_ts)

        return_info = dict(
            recon_cs=recon_cs,
            recon_ts=recon_ts
        )

        return return_info


    def forward(self,
                sample: torch.FloatTensor,
                label: torch.LongTensor = None,
                training: bool = True,
                ):

        encoder_output = self.encode(sample)

        enc_cs = encoder_output["enc_cs"]
        enc_ts = encoder_output["enc_ts"]
        embed_ind_cs = encoder_output["embed_ind_cs"]
        embed_ind_ts = encoder_output["embed_ind_ts"]
        quantized_cs = encoder_output["quantized_cs"]
        quantized_ts = encoder_output["quantized_ts"]
        mask_cs = encoder_output["mask_cs"]
        mask_ts = encoder_output["mask_ts"]
        id_restore_cs = encoder_output["id_restore_cs"]
        id_restore_ts = encoder_output["id_restore_ts"]
        clip_loss = encoder_output["clip_loss"]
        weighted_quantized_loss = (encoder_output["weighted_quantized_loss_cs"] +
                                   encoder_output["weighted_quantized_loss_ts"])
        weighted_commit_loss = (encoder_output["weighted_commit_loss_cs"] +
                                 encoder_output["weighted_commit_loss_ts"])
        weighted_codebook_diversity_loss = (encoder_output["weighted_codebook_diversity_loss_cs"] +
                                            encoder_output["weighted_codebook_diversity_loss_ts"])
        weighted_orthogonal_reg_loss = (encoder_output["weighted_orthogonal_reg_loss_cs"] +
                                        encoder_output["weighted_orthogonal_reg_loss_ts"])

        factors_cs = torch.cat([enc_cs, quantized_cs], dim=-1)
        factors_ts = torch.cat([enc_ts, quantized_ts], dim=-1)

        factors = torch.concat([factors_cs, factors_ts], dim=1)

        posterior = self.encode_post_distribution(factors, label)
        mu_post, sigma_post = posterior.mean, posterior.std
        prior = self.encode_prior_distribution(factors)
        mu_prior, sigma_prior = prior.mean, prior.std

        if training:
            pred_label = self.decode_post_distribution(factors, mu_post, sigma_post)

            decoder_output = self.decode(quantized_cs,
                                         id_restore_cs,
                                         quantized_ts,
                                         id_restore_ts)
        else:
            pred_label = self.decode_post_distribution(factors, mu_prior, sigma_prior)

            decoder_output = self.decode(quantized_cs,
                                         id_restore_cs,
                                         quantized_ts,
                                         id_restore_ts)

        recon_cs = decoder_output["recon_cs"]
        recon_ts = decoder_output["recon_ts"]

        return_info = dict(
            factors_cs=factors_cs,
            factors_ts=factors_ts,
            recon_cs=recon_cs,
            recon_ts=recon_ts,
            embed_ind_cs=embed_ind_cs,
            embed_ind_ts=embed_ind_ts,
            mask_cs=mask_cs,
            mask_ts=mask_ts,
            id_restore_cs=id_restore_cs,
            id_restore_ts=id_restore_ts,
            pred_label=pred_label,
            posterior=posterior,
            prior=prior,
            weighted_clip_loss=clip_loss,
            weighted_quantized_loss=weighted_quantized_loss,
            weighted_commit_loss=weighted_commit_loss,
            weighted_codebook_diversity_loss=weighted_codebook_diversity_loss,
            weighted_orthogonal_reg_loss=weighted_orthogonal_reg_loss
        )

        return return_info


if __name__ == '__main__':
    device = torch.device("cpu")

    cs_embed_config = dict(
        type='PatchEmbed',
        data_size=(64, 29, 152),
        patch_size=(1, 29, 152),
        input_channel=1,
        input_dim=152,
        output_dim=128,
        latent_dim=128,
    )

    ts_embed_config = dict(
                type='PatchEmbed',
                data_size=(64, 29, 152),
                patch_size=(4, 1, 152),
                input_channel=1,
                input_dim=152,
                output_dim=128,
                latent_dim=128,
            )

    cs_config = dict(
        cs_encoder_config = dict(
            type = "TransformerEncoder",
            embed_config=cs_embed_config,
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
        ),
        cs_quantizer_config = dict(
            type="VectorQuantizer",
            dim=128,
            codebook_size=512,
            codebook_dim=128,
            decay=0.99,
            commitment_weight=1.0
        ),
        cs_decoder_config = dict(
            type='TransformerDecoder',
            embed_config=cs_embed_config,
            input_dim=128,
            latent_dim=128,
            output_dim=4,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
        )
    )

    ts_config = dict(
        ts_encoder_config=dict(
            type="TransformerEncoder",
            embed_config=ts_embed_config,
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
        ),
        ts_quantizer_config=dict(
            type="VectorQuantizer",
            dim=128,
            codebook_size=512,
            codebook_dim=128,
            decay=0.99,
            commitment_weight=1.0
        ),
        ts_decoder_config=dict(
            type='TransformerDecoder',
            embed_config=ts_embed_config,
            input_dim=128,
            latent_dim=128,
            output_dim=4,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            cls_embed=True,
            sep_pos_embed=True,
            trunc_init=False,
            no_qkv_bias=False,
        )
    )

    multi_scale_encoder_config = dict(
        depth=2,
        sm_dim=128,
        lg_dim=128,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=16,
        sm_enc_params=dict(
            depth=1,
            heads=8,
            mlp_dim=128,
            dim_head=16
        ),
        lg_enc_params=dict(
            depth=1,
            heads=8,
            mlp_dim=128,
            dim_head=16
        ),
        dropout=0.0
    )

    model = DynamicDualVQVAE(
        cs_embed_config=cs_embed_config,
        ts_embed_config=ts_embed_config,
        cs_config=cs_config,
        ts_config=ts_config,
        multi_scale_encoder_config = multi_scale_encoder_config,
        cl_loss_weight=1.0,
        temperature=1.0
    )

    feature = torch.randn(4, 64, 29, 149)
    temporal = torch.zeros(4, 64, 29, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    label = torch.randn(4, 1, 29)  # batch, next returns, asset nums

    output = model(batch, label)
    print(output["recon_cs"].shape)
    print(output["recon_ts"].shape)
    print(output["pred_label"].shape)

    loss = output["weighted_clip_loss"]
    loss.backward()