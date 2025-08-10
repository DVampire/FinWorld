import torch
import torch.nn as nn
from typing import Dict

from torch.nn import functional as F
from torch.distributions import Normal
from finworld.registry import MODEL

# todo HireVAE主模型未修改，Encoder Extractor等可能需要合并
class Extractor_Stock(nn.Module):
    def __init__(self, extractor_config: Dict):
        super(Extractor_Stock, self).__init__()

        self.input_size = extractor_config["input_features"]
        self.hidden_dim = extractor_config["hidden_dim"]

        self.MLP = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.LeakyReLU()
        )
        self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=False)

    def forward(self, x):
        '''
            input X [b, T, N, C]:
                b: batch size
                T: window size
                N: stock num
                C: feature num
            output e_s [b, N, H]:
                H: hidden_dim
        '''
        batch_num, window_size, stock_num, feature_size = x.shape
        x = torch.permute(x, (1, 0, 2, 3)).reshape(window_size, -1,
                                                   feature_size)  # [window_size, stock_num, feature_size]

        h_proj = self.MLP(x)  # [window_size, stock_num, hidden_dim]

        out, hidden = self.rnn(h_proj)

        e_s = hidden.view((-1, stock_num, self.hidden_dim))  # [batch, stock, hidden_dim]

        return e_s

class Extractor_Market(nn.Module):
    def __init__(self, extractor_config):
        super(Extractor_Market, self).__init__()
        self.device = "cuda"
        self.d = extractor_config["dimension"]
        self.hidden_dim = extractor_config["hidden_dim"]

        self.query = nn.Linear(1, self.hidden_dim)
        self.key = nn.Linear(1, self.hidden_dim)
        self.value = nn.Linear(1, self.hidden_dim)
        self.scale = self.hidden_dim ** -0.5
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.linear2 = nn.Linear(d*(d-1), self.hidden_dim)

        self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)


    def get_cross_attention(self, f1, f2):
        '''
            f1, f2: [b, T, 1]
        '''
        b = f1.size(0)
        n = f1.size(1)
        # [b, T, H]
        query_norm = F.normalize(self.query(f1), p=2, dim=1, eps=1e-12)
        key_norm = F.normalize(self.key(f2), p=2, dim=1, eps=1e-12)
        value_norm = F.normalize(self.value(f2),p=2, dim=1, eps=1e-12)

        # [b, H, n, 1]
        queries = query_norm.view(b, n, self.hidden_dim, -1).transpose(1, 2)
        keys = key_norm.view(b, n, self.hidden_dim, -1).transpose(1, 2)
        values = value_norm.view(b, n, self.hidden_dim, -1).transpose(1, 2)

        # [b, H, n, n]
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        attn = dots.softmax(dim=-1)
        # [b, H, n, 1]
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        # [b, n, H]
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out = self.linear(out)
        return out

    def forward(self, M, d):
        '''
            这里默认 每种index就是一个feature
            input M [b, T, C]
                C: total features of d modalities
            output v [b, T, H]
        '''
        v = None
        for i in range(d):
            for j in range(d):
                if i != j:    # d * (d-1)
                    f1 = M[:,:,i:i+1]
                    f2 = M[:,:,j:j+1]
                    attn = self.get_cross_attention(f1, f2)
                    if v is None:
                        v = attn
                    else:
                        v = torch.cat([v, attn], dim=2)
        # [b, n, d!]
        v = self.linear2(v)

        result, _ = self.rnn(v)

        return result[:,-1,:]

class Encoder_Market(nn.Module):

    def __init__(self, encoder_config):
        super(Encoder_Market, self).__init__()
        self.hidden_dim = encoder_config["hidden_dim"]
        self.factor_num = encoder_config["factor_num"]
        self.beta =encoder_config["beta"]

        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.factor_num),
            nn.LeakyReLU()
        )
        self.sigma = nn.Sequential(
            nn.Linear(self.hidden_dim, self.factor_num),
            nn.LeakyReLU()
            )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dim//2, out_features=self.hidden_dim)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU()
            )
        self.linear2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.factor_num),
            nn.Softplus()
            )

        self.softplus = nn.Softplus()

    def switch(self, v, beta):

        # [b, h]
        score = self.sigma(v)

        mu = self.mu(v)
        sigma = self.softplus(self.sigma(v))
        mu_sort, _ = mu.sort(dim=1, descending=True)
        sigma_sort, _ = sigma.sort(dim=1, descending=True)

        mu_r = beta * mu + (1 - beta) * mu_sort
        sigma_r = self.softplus(beta * sigma + (1 - beta) * sigma_sort)

        dist = Normal(mu_r, sigma_r)    # 高斯分布 N ~ [mu_r, sigma_r]
        # [b, n, h]
        prob_log = dist.log_prob(score)

        c = torch.argmax(prob_log)
        return c    # tensor

    def forward(self, v, y=None, **kwargs):
        '''
            input: v, y
                v:   [b, H]
                y:   [b, N] time x stock_num
            output:
                post_market_mu      [b, Factor]
                post_market_sigma   [b, Factor]
        '''
        batch_size, _ = v.shape
        if y is not None:
            # y_mean [b, H]
            y_mean = self.linear_layer(torch.mean(y, dim=1).unsqueeze(-1))
            y_pred = self.linear(torch.cat([y_mean, v], dim=1))
            y_pred = self.softplus(y_pred)

            post_market_mu = self.mu(y_pred)
            post_market_sigma = self.softplus(F.normalize(self.sigma(y_pred)))

            market = reparameterize(post_market_mu, post_market_sigma)  # [b, factor]

            c = self.switch(v, self.beta)

        else:
            market = self.linear2(v)
            # post_market_mu = self.linear2(v)
            # post_market_sigma = torch.zeros(batch_size, self.factor_num).to("cuda")
            # market = reparameterize(post_market_mu, post_market_sigma)
            c = self.switch(v, self.beta)


        return market, c    # market[batch, window, stock, hidden_dim]

class Encoder_Stock(nn.Module):
    def __init__(self, encoder_config):
        super(Encoder_Stock, self).__init__()
        self.hidden_dim = encoder_config["hidden_dim"]
        self.factor_num = encoder_config["factor_num"]
        
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim)
        )

        self.linear = nn.Linear(in_features=2*self.hidden_dim, out_features=self.hidden_dim)
        self.linear2 = nn.Linear(in_features=self.hidden_dim, out_features=1)

        self.sigma = nn.Linear(in_features=self.hidden_dim, out_features=self.factor_num)
        self.mu = nn.Linear(in_features=self.hidden_dim, out_features=self.factor_num)

        self.softplus = nn.Softplus()


    def forward(self, e_s, v, y=None, **kwargs):
        '''
            input: e_s, y, v
                e_s: [b, N, H]  stock_num x hidden_size
                v:   [b, H]
                y:   [b, N] batch_size x stock_num
            output:
                post_stock_mu        [b, Factor]
                post_stock_sigma     [b, Factor]
        '''
        batch, stock_num, _ = e_s.shape
        if y is not None:
            attn = self.softplus(self.linear_layer(e_s).permute(0, 2, 1))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
            attn = F.normalize(torch.matmul(attn, y.unsqueeze(2)).view(batch, -1))
        else:
            attn = self.softplus(e_s.mean(dim=1))

        y_pred = torch.cat([attn, v], dim=-1)
        y_pred = self.softplus(self.linear(y_pred))

        post_stock_mu = F.normalize(self.mu(y_pred))
        post_stock_sigma = self.softplus(post_stock_mu)

        stock = reparameterize(post_stock_mu, post_stock_sigma)
        return stock    # stock [batch, window, stock, hidden_dim]


class Decoder(nn.Module):

    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        self.hidden_dim = decoder_config["hidden_dim"]
        self.factor_num = decoder_config["factor_num"]


        self.linear = nn.Sequential(
            nn.Linear(in_features=2 * self.factor_num, out_features=self.factor_num),
            nn.Linear(in_features=self.factor_num, out_features=self.hidden_dim),
            nn.LeakyReLU()
            )

        self.mu_layer = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Softplus()
        )

        self.sigma_layer = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Softplus()
        )

        self.proj = nn.Sequential(nn.Linear(
            in_features=self.factor_num, out_features=self.factor_num),
            nn.LeakyReLU()
            )
        self.softplus = nn.Softplus()

    def forward(self, z, m, e_s, c):
        '''
        input:
            z, m: [b, Factor]
            e_s: [b, N, H]
            c: tensor(x)
        output: x [b, N]
        '''
        # c 不确定怎么用进去
        value = c.item()

        z_m = self.linear(torch.cat([self.proj(z), self.proj(m)], dim=1))
        features = self.softplus(value * torch.bmm(e_s, z_m.unsqueeze(-1)))
        features = torch.where(torch.isnan(features), torch.full_like(features, 0), features)

        y_mu = self.mu_layer(features).squeeze(-1)
        y_sigma = F.normalize(self.mu_layer(features)).squeeze(-1)

        return y_mu, y_sigma

@MODEL.register_module(force=True)
class HireVAE(nn.Module):
    def __init__(self, factor_num, input_size, hidden_size, beta, gamma, d):
        super(HireVAE, self).__init__(module_id=-1)     # module_id=-1

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.factor_num = factor_num
        self.beta = beta
        self.gamma = gamma
        self.d = d

        self.prior_mu_layer = nn.Linear(self.hidden_dim, 1)
        self.prior_sigma_layer = nn.Linear(self.hidden_dim, 1)
        self.post_mu_layer = nn.Linear(self.factor_num, self.factor_num)
        self.post_sigma_layer = nn.Linear(self.factor_num, self.factor_num)

        self.extractor_stock = Extractor_Stock(input_size=self.input_dim, hidden_size=self.hidden_dim)
        self.extractor_market = Extractor_Market(hidden_size=self.hidden_dim, d=self.d)
        self.encoder_market = Encoder_Market(factor_num=self.factor_num, hidden_size=self.hidden_dim, beta=self.beta)
        self.encoder_stock = Encoder_Stock(factor_num=self.factor_num, hidden_size=self.hidden_dim)
        self.decoder = Decoder(factor_num=self.factor_num, hidden_size=self.hidden_dim)
        # self.predictor = Predictor(factor_num=self.factor_num, hidden_size=self.hidden_dim)

        self.leakyrelu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.concat_layer = nn.Linear(2*self.factor_num, self.factor_num)

    def forward(self, X, M, y=None, **kwargs):
        '''
            input: X, M, y
                X: [batch, stock_num, features]
                M: [batch, T, d]
                y: [batch, stock_num, 1]
        '''
        d = M.size(-1)
        # ========= feature extractors
        stock_features = self.extractor_stock(X)
        market_features = self.extractor_market(M, d)

        if y is not None:
            # ========= encoders
            market, condition = self.encoder_market(market_features, y)
            stock = self.encoder_stock(stock_features, market_features, y)

            latent_features = self.concat_layer(torch.concat([market, stock], dim=1))

            z_post_mu = self.post_mu_layer(latent_features)
            z_post_sigma = self.softplus(F.normalize(self.post_sigma_layer(latent_features)))

        else:
            # ========= decoders
            market, condition = self.encoder_market(market_features)
            stock = self.encoder_stock(stock_features, market_features)
            z_post_mu, z_post_sigma = None, None

        # ==========================
        prediction_mu, prediction_sigma = self.decoder(stock, market, stock_features, condition)

        return prediction_mu, prediction_sigma, z_post_mu, z_post_sigma


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z

if __name__ == "__main__":
    # stock_num = 10, features = 5
    d = 3
    beta = 0.5
    x = torch.randn(16, 2, 10, 5)
    M = torch.randn(16, 2, d)
    y = torch.randn(16, 10)

    # extractor_stock = Extractor_Stock(input_size=5, hidden_size=16)
    # # [b, N, H]
    # e_s = extractor_stock(x)
    #
    # extractor_market = Extractor_Market(hidden_size=16)
    # v = extractor_market(M,d)
    #
    # encoder_market = Encoder_Market(hidden_size=16, beta=beta)
    # market_sample, c = encoder_market(v, y)
    #
    # encoder_stock = Encoder_Stock(hidden_size=16)
    # stock_sample = encoder_stock(e_s, v, y)
    #
    # decoder = Decoder(hidden_size=16)
    # x_pred = decoder(stock_sample, market_sample, e_s, c)



    model = HireVAE(factor_num=16, input_size=5, hidden_size=64, beta=beta, alpha=1, d=d)
    result = model(x, M, y)
    print(result.shape)
