import torch
import torch.nn as nn
from typing import Dict
from torch.nn import functional as F
from einops import rearrange
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import MODEL
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import QUANTIZER
from finworld.registry import EMBED
from finworld.models.modules.transformer import Mlp
from finworld.models.modules.distribution import DiagonalGaussianDistribution


@MODEL.register_module(force=True)
class DynamicSingleVQVAE(nn.Module):
    def __init__(self,
                 embed_config: Dict = None,  # cross-sectional embedding config
                 config: Dict = None,  # cross-sectional config
                 asset_num: int = 29,  # asset number
                 ):
        super(DynamicSingleVQVAE, self).__init__()

        self.asset_num = asset_num

        self.encoder_config = config.get("encoder_config", {})  # cross-sectional factor encoder config
        self.quantizer_config = config.get("quantizer_config", {})  # cross-sectional factor quantizer config
        self.decoder_config = config.get("decoder_config", {})  # cross-sectional reconstruction decoder config

        self.embed_layer = EMBED.build(embed_config)
        self.encoder = ENCODER.build(self.encoder_config)
        self.quantizer = QUANTIZER.build(self.quantizer_config)
        self.decoder = DECODER.build(self.decoder_config)
        self.if_mask = self.encoder.if_mask
        self.patch_size = self.embed_layer.patch_size
        self.output_dim = self.decoder.output_dim

        latent_dim = self.encoder.latent_dim * 2
        factor_num = self.encoder.n_num
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

        embed = self.embed_layer(sample)

        enc, mask, id_restore = self.encoder(embed)

        quantized, embed_ind, quantized_loss, quantized_loss_breakdown = self.quantizer(enc)
        weighted_quantized_loss = quantized_loss[0]
        weighted_commit_loss = quantized_loss_breakdown.weighted_commit_loss
        weighted_codebook_diversity_loss = quantized_loss_breakdown.weighted_codebook_diversity_loss
        weighted_orthogonal_reg_loss = quantized_loss_breakdown.weighted_orthogonal_reg_loss

        return_info = dict(
            enc=enc,
            quantized=quantized,
            embed_ind=embed_ind,
            mask=mask,
            id_restore=id_restore,
            weighted_quantized_loss=weighted_quantized_loss,
            weighted_commit_loss=weighted_commit_loss,
            weighted_codebook_diversity_loss=weighted_codebook_diversity_loss,
            weighted_orthogonal_reg_loss=weighted_orthogonal_reg_loss
        )

        return return_info

    @apply_forward_hook
    def encode_post_distribution(self,
                                 factors: torch.Tensor,
                                 label: torch.FloatTensor = None):

        label = rearrange(label, 'n c t s-> (n c t) s')
        label = label.unsqueeze(-1)

        portfolio_weights = self.to_pd_encode_latent(factors)
        portfolio_weights = F.softmax(portfolio_weights, dim=-1)
        returns = torch.matmul(portfolio_weights, label).squeeze(-1)
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
               quantized: torch.FloatTensor,
               ids_restore: torch.LongTensor, ):

        recon = self.decoder(quantized, ids_restore=ids_restore)

        return_info = dict(
            recon=recon,
        )

        return return_info

    def forward(self,
                sample: torch.FloatTensor,
                label: torch.LongTensor = None,
                training: bool = True,
                ):

        encoder_output = self.encode(sample)

        enc = encoder_output["enc"]
        quantized = encoder_output["quantized"]
        mask = encoder_output["mask"]
        id_restore = encoder_output["id_restore"]
        embed_ind = encoder_output["embed_ind"]
        weighted_quantized_loss = encoder_output["weighted_quantized_loss"]
        weighted_commit_loss = encoder_output["weighted_commit_loss"]
        weighted_codebook_diversity_loss = encoder_output["weighted_codebook_diversity_loss"]
        weighted_orthogonal_reg_loss = encoder_output["weighted_orthogonal_reg_loss"]

        factors = torch.concat([enc, quantized], dim=-1)

        posterior = self.encode_post_distribution(factors, label)
        mu_post, sigma_post = posterior.mean, posterior.std
        prior = self.encode_prior_distribution(factors)
        mu_prior, sigma_prior = prior.mean, prior.std

        if training:
            pred_label = self.decode_post_distribution(factors, mu_post, sigma_post)

            decoder_output = self.decode(quantized, id_restore)
        else:
            pred_label = self.decode_post_distribution(factors, mu_prior, sigma_prior)

            decoder_output = self.decode(quantized, id_restore)

        recon = decoder_output["recon"]

        return_info = dict(
            factors=factors,
            embed_ind=embed_ind,
            recon=recon,
            mask=mask,
            id_restore=id_restore,
            pred_label=pred_label,
            posterior=posterior,
            prior=prior,
            weighted_quantized_loss=weighted_quantized_loss,
            weighted_commit_loss=weighted_commit_loss,
            weighted_codebook_diversity_loss=weighted_codebook_diversity_loss,
            weighted_orthogonal_reg_loss=weighted_orthogonal_reg_loss
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

    config = dict(
        encoder_config=dict(
            type="VAETransformerEncoder",
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
        ),
        quantizer_config=dict(
            type="VectorQuantizer",
            dim=128,
            codebook_size=512,
            codebook_dim=128,
            decay=0.99,
            commitment_weight=1.0
        ),
        decoder_config=dict(
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
    )

    model = DynamicSingleVQVAE(
        embed_config=embed_config,
        config=config,
    )

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    batch = torch.cat([feature, temporal], dim=-1).to(device)
    label = torch.randn(4, 1, 1, 29)  # batch, channel, next returns, asset nums

    output = model(batch, label)
    print(output["recon"].shape)



