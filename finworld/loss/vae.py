import torch
from torch import nn

from finworld.registry import LOSS
from finworld.loss.base import Loss

@LOSS.register_module(force=True)
class VAELoss(Loss):
    def __init__(
        self,
        logvar_init = 0.0,
        kl_loss_weight = 0.000001,
        nll_loss_weight = 1.0,
    ):
        super().__init__()
        self.logvar_init = logvar_init
        self.kl_loss_weight = kl_loss_weight
        self.nll_loss_weight = nll_loss_weight
        # KL Loss
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def __str__(self):
        return f"VAELoss(logvar_init={self.logvar_init}, kl_loss_weight={self.kl_loss_weight}, nll_loss_weight={self.nll_loss_weight})"

    def forward(
        self,
        sample,
        target_sample,
        posterior,
        mask = None,
        if_mask = False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param posterior: DiagonalGaussianDistribution
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert sample.shape == target_sample.shape

        # reconstruction loss
        recon_loss = torch.abs(target_sample - sample)

        nll_loss = recon_loss / torch.exp(self.logvar) + self.logvar

        if if_mask:
            mask = mask.repeat(1, 1, nll_loss.shape[-1])
            nll_loss = nll_loss * mask

        # Weighted NLL Loss
        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        # NLL Loss
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # Weighted KL Loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = kl_loss * self.kl_loss_weight

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_kl_loss=weighted_kl_loss,
        )

        return loss_dict