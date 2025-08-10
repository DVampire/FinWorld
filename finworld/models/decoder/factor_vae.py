import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


from finworld.registry import DECODER
from finworld.registry import EMBED
from finworld.models.decoder.base import Decoder
from finworld.models.modules.transformer import Mlp
from finworld.models.encoder import FactorVAEEncoder

@DECODER.register_module(force=True)
class FactorVAEDecoder(Decoder):
    def __init__(self,
                 embed_config: dict = None,
                 input_dim: int = 128,
                 latent_dim: int = 128,
                 portfolio_num: int = 20,
                 factor_num: int = 32,
                 trunc_init: bool = False
                 ):
        super(FactorVAEDecoder, self).__init__()

        self.data_size = to_2tuple(embed_config.get('data_size', None))
        self.patch_size = to_2tuple(embed_config.get('patch_size', None))

        self.input_size = (
            self.data_size[0] // self.patch_size[0],
            self.data_size[1] // self.patch_size[1],
            self.data_size[2] // self.patch_size[2]
        )
        self.num_patches = self.input_size[0] * self.input_size[1] * self.input_size[2]

        self.trunc_init = trunc_init
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.alpha_hidden_layer = Mlp(
            in_features=input_dim,
            hidden_features=latent_dim // 2,
            out_features=latent_dim,
            act_layer=nn.LeakyReLU
        )

        self.alpha_mu_layer = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim // 2,
            out_features=1,
            act_layer=nn.LeakyReLU
        )

        self.alpha_sigma_layer = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim // 2,
            out_features=1,
            act_layer=nn.LeakyReLU
        )
        self.softplus = nn.Softplus()

        self.beta_layer = Mlp(
            in_features=input_dim,
            hidden_features=latent_dim // 2,
            out_features=factor_num,
            act_layer=nn.LeakyReLU
        )

        self.initialize_weights()

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, z_mu, z_sigma, e):
        """
        :param z_mu: post mean, [batch, factor_num, 1]
        :param z_sigma: post sigma, [batch, factor_num, 1]
        :param e: Latent features, [batch, stock_num, hidden_dim]
        :return
        """

        batch, stock_num, _ = e.shape
        # alpha layer
        # alpha_h = [batch, stock_num, hidden_dim]
        alpha_h = self.alpha_hidden_layer(e)
        # alpha_mu = [batch, stock_num, 1]
        alpha_mu = self.alpha_mu_layer(alpha_h)
        # alpha_sigma = [batch, stock_num, 1]
        alpha_sigma = self.alpha_sigma_layer(alpha_h)
        alpha_sigma = self.softplus(alpha_sigma)

        # beta layer
        # beta = [batch, stock_num, factor_num]
        beta = self.beta_layer(e)

        # y_mu = [batch, stock_num, 1]
        y_mu = torch.bmm(beta, z_mu) + alpha_mu

        alpha_sigma_pow = alpha_sigma.pow(2)
        beta_pow = beta.pow(2)
        z_sigma_pow = z_sigma.pow(2)

        z_sigma_pow = torch.bmm(beta_pow, z_sigma_pow)

        y_sigma = torch.sqrt(alpha_sigma_pow + z_sigma_pow)

        sample = self.reparameterize(y_mu, y_sigma)
        sample = sample.squeeze(-1)

        return sample

if __name__ == '__main__':
    device = torch.device("cpu")

    embed_config = dict(
        type='FactorVAEEmbed',
        data_size=(64, 29, 152),
        patch_size=(64, 1, 152),
        input_dim=152,
        input_channel=1,
        temporal_dim=3,
        embed_dim=128)

    embed_layer = EMBED.build(embed_config).to(device)

    encoder = FactorVAEEncoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        portfolio_num=20,
        factor_num=32).to(device)

    decoder = FactorVAEDecoder(
        embed_config=embed_config,
        input_dim=128,
        latent_dim=128,
        portfolio_num=20,
        factor_num=32).to(device)

    feature = torch.randn(4, 1, 64, 29, 149)
    temporal = torch.zeros(4, 1, 64, 29, 3)
    label = torch.randn(4, 1, 1, 29)  # batch, channel, next returns, asset nums

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    embed = embed_layer(batch)
    mu_post, sigma_post = encoder(embed, label)

    sample = decoder(mu_post, sigma_post, embed)
    print(sample.shape)



