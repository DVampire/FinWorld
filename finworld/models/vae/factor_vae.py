import torch
import torch.nn as nn
from typing import Dict
from diffusers.utils.accelerate_utils import apply_forward_hook

from finworld.registry import EMBED
from finworld.registry import ENCODER
from finworld.registry import DECODER
from finworld.registry import MODEL
from finworld.registry import PREDICTOR

@MODEL.register_module(force=True)
class FactorVAE(nn.Module):
    def __init__(self,
                 embed_config: Dict,
                 encoder_config: Dict,
                 decoder_config: Dict,
                 predictor_config: Dict,
                 gamma = 1.0,
                 ):
        super().__init__()

        self.gamma = gamma
        self.embed = EMBED.build(embed_config)
        self.encoder = ENCODER.build(encoder_config)
        self.decoder = DECODER.build(decoder_config)
        self.predictor = PREDICTOR.build(predictor_config)

    def get_decoder_distribution(
        self, mu_alpha, sigma_alpha, mu_factor, sigma_factor, beta
    ):
        # print(mu_alpha.shape, mu_factor.shape, sigma_factor.shape, beta.shape)
        mu_dec = mu_alpha + torch.bmm(beta, mu_factor)

        sigma_dec = torch.sqrt(
            torch.square(sigma_alpha)
            + torch.bmm(torch.square(beta), torch.square(sigma_factor))
        )

        return mu_dec, sigma_dec

    @apply_forward_hook
    def encode(self, latent_features, label):
        mu, sigma = self.encoder(latent_features, label)

        return_info = dict(
            mu = mu,
            sigma = sigma
        )

        return return_info

    @apply_forward_hook
    def decode(self, mu_post: torch.FloatTensor,
               sigma_post: torch.FloatTensor,
               latent_features: torch.FloatTensor):

        sample = self.decoder(mu_post, sigma_post, latent_features)

        return_info = dict(
            sample = sample
        )

        return return_info

    def forward(self,
                sample: torch.FloatTensor,
                label: torch.LongTensor = None,
                training: bool = True,
                ):
        latent_features = self.embed(sample)

        encoder_output = self.encode(latent_features, label)

        mu_post = encoder_output["mu"]
        sigma_post = encoder_output["sigma"]

        decoder_output = self.decode(mu_post, sigma_post, latent_features)
        sample = decoder_output["sample"]

        mu_pred, sigma_pred = self.predictor(latent_features)

        return_info = dict(
            sample = sample,
            mu_post = mu_post,
            sigma_post = sigma_post,
            mu_pred = mu_pred,
            sigma_pred = sigma_pred
        )

        return return_info



if __name__ == "__main__":
    device = torch.device("cpu")

    embed_config = dict(
        type='FactorVAEEmbed',
        data_size=(64, 29, 152),
        patch_size=(64, 1, 152),
        input_dim=152,
        input_channel=1,
        temporal_dim=3,
        embed_dim=128)

    encoder_config = dict(
        type='FactorVAEEncoder',
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        portfolio_num=20,
        factor_num=32
    )

    decoder_config = dict(
        type='FactorVAEDecoder',
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        portfolio_num=20,
        factor_num=32
    )

    predictor_config = dict(
        type='FactorVAEPredictor',
        input_dim=128,
        latent_dim=128,
        factor_num=32
    )

    model = FactorVAE(
        embed_config=embed_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        predictor_config=predictor_config
    ).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    label = torch.randn(4, 1, 1, 29) # batch, channel, next returns, asset nums
    batch = torch.cat([feature, temporal], dim=-1).to(device)

    output = model(batch, label)
    print(output["sample"].shape)
