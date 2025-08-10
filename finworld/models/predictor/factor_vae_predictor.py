import torch
import torch.nn as nn
import torch.nn.functional as F


from finworld.registry import PREDICTOR
from finworld.models.modules.transformer import Mlp

@PREDICTOR.register_module(force=True)
class FactorVAEPredictor(nn.Module):
    def __init__(self,
                 input_dim = 128,
                 latent_dim = 128,
                 factor_num = 32,
                 portfolio_num = 20):
        super(FactorVAEPredictor, self).__init__()

        self.key_layer = nn.ModuleList([nn.Linear(input_dim, latent_dim) for _ in range(factor_num)])
        self.value_layer = nn.ModuleList([nn.Linear(input_dim, latent_dim) for _ in range(factor_num)])
        self.query = nn.Parameter(torch.randn(factor_num, latent_dim))

        self.dist_mu = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim // 2,
            out_features=1,
            act_layer=nn.LeakyReLU
        )

        self.dist_sigma = Mlp(
            in_features=latent_dim,
            hidden_features=latent_dim // 2,
            out_features=1,
            act_layer=nn.LeakyReLU
        )

        self.softplus = nn.Softplus()

        self.factor_num = factor_num
    def forward(self, latent_features):
        """
        :param e: Latent features, [batch, stock_num, hidden_dim]
        :return: [batch, factor_nums]
        """

        batch_num, stock_num, hidden_dim = latent_features.shape

        h_multi = torch.zeros(batch_num, self.factor_num, hidden_dim).to(latent_features.device)

        for factor in range(self.factor_num):
            # k = [batch, stock_num, hidden_dim]
            # v = [batch, stock_num, hidden_dim]
            k = self.key_layer[factor](latent_features)
            v = self.value_layer[factor](latent_features)

            # attn = [batch, stock_num]
            attn = torch.matmul(k, self.query[factor]).view(batch_num, stock_num)

            # q_norm2 = [1]
            q_norm2 = torch.sqrt(self.query[factor].pow(2).sum())
            # k_norm2 = [batch, stock_num]
            k_norm2 = torch.sqrt(k.pow(2).sum(dim=2))

            # norm2 = [batch, stock_num]
            norm2 = k_norm2 * q_norm2

            # attn = [batch, stock_num]
            attn = torch.div(attn, norm2)

            # attn_sum = [batch]
            attn_sum = torch.sum(attn, dim=1)

            # attn = [batch, 1, stock_num]
            attn = torch.div(attn, attn_sum.view(batch_num, 1)).unsqueeze(1)

            h_att = torch.bmm(attn, v).view(batch_num, hidden_dim)

            h_multi[:, factor, :] = h_att

        # h_multi = [batch_num, factor_num, hidden_dim]
        prior_mu = self.dist_mu(h_multi)

        prior_sigma = self.dist_sigma(h_multi)
        prior_sigma = self.softplus(prior_sigma) + 1e-6

        return prior_mu, prior_sigma    # [b, factor, 1]

if __name__ == '__main__':
    device = torch.device("cpu")

    model = FactorVAEPredictor(
        input_dim=128,
        latent_dim=128,
        factor_num=32
    ).to(device)


    batch = torch.randn(4, 29, 128).to(device)

    mu, sigma = model(batch)

    print(mu.shape, sigma.shape)