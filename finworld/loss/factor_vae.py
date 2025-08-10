import torch
from torch import nn

from finworld.registry import LOSS

@LOSS.register_module(force=True)
class FactorVAELoss(nn.Module):
    def __init__(self,
                 kl_loss_weight = 0.000001,
                 nll_loss_weight = 1.0,):
        super().__init__()
        self.kl_loss_weight = kl_loss_weight
        self.nll_loss_weight = nll_loss_weight

    def __str__(self):
        return f"FactorVAELoss(kl_loss_weight={self.kl_loss_weight}, nll_loss_weight={self.nll_loss_weight})"

    def forward(
        self,
        sample,
        target_sample,
        mu_post,
        sigma_post,
        mu_pred,
        sigma_pred
    ):
        assert sample.shape == target_sample.shape

        rec_loss = (sample - target_sample) ** 2
        nll_loss = rec_loss

        # Weighted NLL Loss
        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        kl_loss = (torch.log(sigma_pred / sigma_post) + (sigma_post**2 + (mu_post - mu_pred)**2) / (2 * sigma_pred **2) - 0.5)
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = kl_loss * self.kl_loss_weight
        weighted_kl_loss = torch.sum(weighted_kl_loss) / weighted_kl_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_kl_loss=weighted_kl_loss,
        )

        return loss_dict