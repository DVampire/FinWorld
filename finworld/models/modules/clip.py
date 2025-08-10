import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import numpy as np

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

class CLIPLayer(nn.Module):
    def __init__(self, temperature=1., cl_loss_weight = 1.0):
        super().__init__()
        self.cl_loss_weight = cl_loss_weight
        self.temperature = float(np.exp(temperature))

    def __str__(self):
        return f"CLIPLoss(temperature={self.temperature}, cl_loss_weight={self.cl_loss_weight})"

    def forward(self, cs_embedding, ts_embedding):
        """
        :param cs_embedding: (N, L, D)
        :param ts_embedding: (N, L, D)
        :return: loss dict
        """

        # use CLS token
        cs_embedding = cs_embedding[:, 0] if cs_embedding.ndim == 3 else cs_embedding
        ts_embedding = ts_embedding[:, 0] if ts_embedding.ndim == 3 else ts_embedding

        cs_embedding, ts_embedding = map(l2norm, (cs_embedding, ts_embedding))

        cs_to_ts = torch.matmul(cs_embedding, ts_embedding.t()) * self.temperature
        ts_to_cs = rearrange(cs_to_ts, '... t i -> ... i t')
        
        cs_to_ts_exp, ts_to_cs_exp = map(torch.exp, (cs_to_ts, ts_to_cs))
        cs_to_ts_pos, ts_to_cs_pos = map(matrix_diag, (cs_to_ts_exp, ts_to_cs_exp))
        cs_to_ts_denom, ts_to_cs_denom = map(lambda t: t.sum(dim = -1), (cs_to_ts_exp, ts_to_cs_exp))

        cs_to_ts_loss = (-log(cs_to_ts_pos) + log(cs_to_ts_denom)).mean(dim = -1)
        ts_to_cs_loss = (-log(ts_to_cs_pos) + log(ts_to_cs_denom)).mean(dim = -1)

        cl_loss = (cs_to_ts_loss + ts_to_cs_loss) / 2.0

        weighted_cl_loss = cl_loss
        if self.cl_loss_weight is not None:
            weighted_cl_loss = self.cl_loss_weight * cl_loss
        weighted_cl_loss = torch.sum(weighted_cl_loss) / weighted_cl_loss.shape[0]
        
        loss_dict = dict(
            cl_loss = cl_loss,
            weighted_cl_loss = weighted_cl_loss
        )
        
        return loss_dict
        
        
