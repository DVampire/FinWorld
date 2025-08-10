import torch
from torch import nn

from finworld.registry import LOSS
from finworld.loss.base import Loss

@LOSS.register_module(force=True)
class VQVAELoss(Loss):
    def __init__(self,
                 nll_loss_weight = 1.0):
        super().__init__()
        self.nll_loss_weight = nll_loss_weight

    def forward(
        self,
        sample,
        target_sample,
        mask = None,
        if_mask = False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert sample.shape == target_sample.shape

        rec_loss = (sample - target_sample) ** 2
        nll_loss = rec_loss

        if if_mask:
            mask = mask.repeat(1, 1, nll_loss.shape[-1])
            nll_loss = nll_loss * mask

        # Weighted NLL Loss
        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
        )

        return loss_dict

@LOSS.register_module(force=True)
class SingleVQVAELoss(Loss):
    def __init__(self,
                 cs_scale = 1.0,
                 nll_loss_weight = 1.0,
                 ret_loss_weight = 1.0,
                 kl_loss_weight = 0.000001):
        super().__init__()
        self.cs_scale = cs_scale
        self.nll_loss_weight = nll_loss_weight
        self.ret_loss_weight = ret_loss_weight
        self.kl_loss_weight = kl_loss_weight

    def forward(
        self,
        sample,
        target_sample,
        label,
        pred_label,
        posterior,
        prior,
        mask=None,
        if_mask=False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert sample.shape == target_sample.shape

        rec_loss = (sample - target_sample) ** 2
        nll_loss = rec_loss

        if if_mask:
            mask = mask.repeat(1, 1, nll_loss.shape[-1])
            nll_loss = nll_loss * mask

        weighted_nll_loss = nll_loss
        if self.nll_loss_weight is not None:
            weighted_nll_loss = self.nll_loss_weight * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        ret_loss = (label - pred_label) ** 2
        ret_loss = ret_loss.mean(dim=-1)

        weighted_ret_loss = ret_loss
        if self.ret_loss_weight is not None:
            weighted_ret_loss = self.ret_loss_weight * ret_loss
        weighted_ret_loss = torch.sum(weighted_ret_loss) / weighted_ret_loss.shape[0]

        kl_loss = posterior.kl(prior, dims=[1])
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = self.kl_loss_weight * kl_loss
        weighted_kl_loss = torch.sum(weighted_kl_loss) / weighted_kl_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_ret_loss=weighted_ret_loss,
            weighted_kl_loss=weighted_kl_loss,
        )

        return loss_dict

@LOSS.register_module(force=True)
class DualVQVAELoss(Loss):
    def __init__(self,
                 cs_scale = 1.0,
                 nll_loss_weight = 1.0,
                 ret_loss_weight = 1.0,
                 kl_loss_weight = 0.000001):
        super().__init__()
        self.cs_scale = cs_scale
        self.nll_loss_weight = nll_loss_weight
        self.ret_loss_weight = ret_loss_weight
        self.kl_loss_weight = kl_loss_weight
    def forward(
        self,
        cs_sample,
        cs_target_sample,
        ts_sample,
        ts_target_sample,
        label,
        pred_label,
        posterior,
        prior,
        cs_mask=None,
        ts_mask=None,
        cs_if_mask=False,
        ts_if_mask=False,
    ):
        """
        :param sample: (N, L, D)
        :param target_sample: (N, L, D)
        :param mask: (N, L)
        :param if_mask: bool
        :return: loss dict
        """

        assert cs_sample.shape == cs_target_sample.shape and ts_sample.shape == ts_target_sample.shape

        cs_rec_loss = (cs_sample - cs_target_sample) ** 2
        cs_nll_loss = cs_rec_loss
        ts_rec_loss = (ts_sample - ts_target_sample) ** 2
        ts_nll_loss = ts_rec_loss

        if cs_if_mask:
            cs_mask = cs_mask.repeat(1, 1, cs_nll_loss.shape[-1])
            cs_nll_loss = cs_nll_loss * cs_mask
        if ts_if_mask:
            ts_mask = ts_mask.repeat(1, 1, ts_nll_loss.shape[-1])
            ts_nll_loss = ts_nll_loss * ts_mask

        # Weighted NLL Loss
        weighted_cs_nll_loss = cs_nll_loss
        if self.nll_loss_weight is not None:
            weighted_cs_nll_loss = self.nll_loss_weight * cs_nll_loss
        weighted_cs_nll_loss = torch.sum(weighted_cs_nll_loss) / weighted_cs_nll_loss.shape[0]
        if self.cs_scale is not None:
            weighted_cs_nll_loss = self.cs_scale * weighted_cs_nll_loss

        weighted_ts_nll_loss = ts_nll_loss
        if self.nll_loss_weight is not None:
            weighted_ts_nll_loss = self.nll_loss_weight * ts_nll_loss
        weighted_ts_nll_loss = torch.sum(weighted_ts_nll_loss) / weighted_ts_nll_loss.shape[0]

        nll_loss = torch.sum(cs_nll_loss) / cs_nll_loss.shape[0] + torch.sum(ts_nll_loss) / ts_nll_loss.shape[0]

        weighted_nll_loss = weighted_cs_nll_loss + weighted_ts_nll_loss

        ret_loss = (label - pred_label) ** 2
        ret_loss = ret_loss.mean(dim=-1)

        weighted_ret_loss = ret_loss
        if self.ret_loss_weight is not None:
            weighted_ret_loss = self.ret_loss_weight * ret_loss
        weighted_ret_loss = torch.sum(weighted_ret_loss) / weighted_ret_loss.shape[0]

        kl_loss = posterior.kl(prior, dims=[1])
        weighted_kl_loss = kl_loss
        if self.kl_loss_weight is not None:
            weighted_kl_loss = self.kl_loss_weight * kl_loss
        weighted_kl_loss = torch.sum(weighted_kl_loss) / weighted_kl_loss.shape[0]

        loss_dict = dict(
            nll_loss=nll_loss,
            weighted_cs_nll_loss=weighted_cs_nll_loss,
            weighted_ts_nll_loss=weighted_ts_nll_loss,
            weighted_nll_loss=weighted_nll_loss,
            weighted_ret_loss=weighted_ret_loss,
            weighted_kl_loss=weighted_kl_loss,
        )

        return loss_dict