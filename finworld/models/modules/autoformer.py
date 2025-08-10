
import math
from typing import Optional, Type
import torch
from torch import nn as nn
from timm.layers import LayerNorm, DropPath
import torch.nn.functional as F

from finworld.models.modules.layerscale import LayerScale


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism.

    This replaces the self-attention family with two phases:

    1. **Period–based dependency discovery** using FFT-based correlation.
    2. **Time-delay aggregation** that gathers the most relevant lags.

    Notes
    -----
    * Follows the implementation described in “Autoformer: Decomposition …”.
    * Variable names mirror the style used in ``finworld.models.modules.attention``.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 1,
        scale: Optional[float] = None,
        attn_drop: float = 0.1,
        output_attn: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        mask_flag :
            If ``True``, apply causal masking (unused here, but kept for parity).
        factor :
            Multiplicative constant for ``top_k = factor * log(seq_len)``.
        scale :
            Optionally scale query before FFT (unused, kept for parity).
        attn_drop :
            Dropout applied to aggregated values.
        output_attn :
            If ``True``, also return raw correlation map.
        """
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.output_attn = output_attn
        self.dropout = nn.Dropout(attn_drop)

    # --------------------------------------------------------------------- #
    #                          Helper – training mode                       #
    # --------------------------------------------------------------------- #
    def _delay_agg_train(self, v: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Aggregation path for *training* (batch-norm style).

        Shapes
        -------
        v     : [B, H, C, L]
        corr  : [B, H, C, L]
        """
        head, chan, length = v.shape[1:]
        top_k = int(self.factor * math.log(length))
        mean_corr = corr.mean(dim=(1, 2))                        # [B, L]
        idx = torch.topk(mean_corr.mean(0), top_k).indices       # [top_k]

        # Soft weights over the selected lags
        weights = torch.stack([mean_corr[:, i] for i in idx], -1)   # [B, top_k]
        alpha = torch.softmax(weights, dim=-1)                      # [B, top_k]

        agg = torch.zeros_like(v)
        for i, lag in enumerate(idx):
            rolled = torch.roll(v, -int(lag), dims=-1)
            agg = agg + rolled * alpha[:, i].view(-1, 1, 1, 1)

        return agg

    # --------------------------------------------------------------------- #
    #                         Helper – inference mode                       #
    # --------------------------------------------------------------------- #
    def _delay_agg_infer(self, v: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """
        Aggregation path for *inference* (index-gather style).

        Shapes
        -------
        v     : [B, H, C, L]
        corr  : [B, H, C, L]
        """
        b, head, chan, length = v.shape
        top_k = int(self.factor * math.log(length))

        base_idx = torch.arange(length, device=v.device)
        base_idx = base_idx.view(1, 1, 1, -1).expand(b, head, chan, -1)  # [B,H,C,L]

        mean_corr = corr.mean(dim=(1, 2))                    # [B, L]
        weights, delay = torch.topk(mean_corr, top_k, dim=-1)  # each [B, top_k]
        alpha = torch.softmax(weights, dim=-1)                 # [B, top_k]

        v_pad = v.repeat_interleave(2, dim=-1)               # length → 2L
        agg = torch.zeros_like(v)

        for i in range(top_k):
            gather_idx = base_idx + delay[:, i].view(-1, 1, 1, 1)
            gathered = torch.gather(v_pad, -1, gather_idx)
            agg = agg + gathered * alpha[:, i].view(-1, 1, 1, 1)

        return agg

    # --------------------------------------------------------------------- #
    #                                Forward                                #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        q: torch.Tensor,  # [B, L, H, D]
        k: torch.Tensor,  # same as q
        v: torch.Tensor,  # [B, S, H, D] (S ≥ L handled internally)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Align seq lengths
        b, l, h, d = q.shape
        s = v.shape[1]
        if l > s:
            pad = torch.zeros_like(q[:, :(l - s), :])
            v = torch.cat([v, pad], 1)
            k = torch.cat([k, pad], 1)
        else:
            v, k = v[:, :l], k[:, :l]

        # Phase 1 – FFT correlation
        q_fft = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)
        k_fft = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
        corr = torch.fft.irfft(q_fft * k_fft.conj(), dim=-1)  # [B,H,D,L]

        # Phase 2 – Delay aggregation
        v_perm = v.permute(0, 2, 3, 1)                        # [B,H,D,L]
        if self.training:
            agg = self._delay_agg_train(v_perm, corr)
        else:
            agg = self._delay_agg_infer(v_perm, corr)

        out = self.dropout(agg).permute(0, 3, 1, 2).contiguous()  # → [B,L,H,D]
        return (out, corr.permute(0, 3, 1, 2)) if self.output_attn else (out, None)

class AutoCorrelationAttention(nn.Module):
    """
    QKV projections + AutoCorrelation fusion (drop-in for Attention).
    Now supports explicit q, k, v input for both self-attn and cross-attn.
    If k/v is None, uses q as default (for self-attn).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        factor: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        output_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Linear projections (B, L, dim) → (B, L, H, head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.corr = AutoCorrelation(
            factor=factor,
            attn_drop=attn_drop,
            output_attn=output_attn,
        )
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,                       # [B, L_q, dim]
        k: Optional[torch.Tensor] = None,      # [B, L_k, dim]
        v: Optional[torch.Tensor] = None,      # [B, L_k, dim]
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, lq, _ = q.shape
        h = self.num_heads
        d = q.size(-1) // h

        # If k or v is None, default to q (self-attn)
        k = q if k is None else k
        v = q if v is None else v

        lk = k.shape[1]

        # Project & reshape
        q_proj = self.q_proj(q).view(b, lq, h, d)
        k_proj = self.k_proj(k).view(b, lk, h, d)
        v_proj = self.v_proj(v).view(b, lk, h, d)

        agg, _ = self.corr(q_proj, k_proj, v_proj, attn_mask)     # [B, L_q, H, D]
        out = agg.view(b, lq, d * h)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MovingAvg(nn.Module):
    """Moving–average smoother (high-lights trend)."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[:2]
        other_dims = x.shape[2:]

        x_flat = x.view(B, L, -1)  # [B, L, C_flat]

        # Pad both ends to keep length
        pad_len = (self.kernel_size - 1) // 2
        front = x_flat[:, 0:1, :].repeat(1, pad_len, 1)
        back = x_flat[:, -1:, :].repeat(1, pad_len, 1)
        x_padded = torch.cat((front, x_flat, back), dim=1)  # [B, L+2p, C_flat]

        x_padded = x_padded.permute(0, 2, 1)  # → [B, C_flat, L+2p]
        trend = self.avg(x_padded)  # [B, C_flat, L+2p-k+1]
        trend = trend.permute(0, 2, 1)  # → [B, L, C_flat]

        trend = trend.view(B, L, *other_dims)
        return trend

class SeriesDecomp(nn.Module):
    """Series decomposition: *x = seasonal + trend*."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AutoformerEncodeBlock(nn.Module):
    """Autoformer encoder layer with progressive decomposition.

    Follows the *TransformerBlock* API so it can be dropped into the same
    scaffolding that expects `norm1 / attn / ls1 / drop_path1 …` structure.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        moving_avg: int = 25,
        qkv_bias: bool = False,
        factor: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        attn_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # --- Attention branch ------------------------------------------------
        self.norm1 = norm_layer(dim)

        if attn_layer is not None:
            self.attn = attn_layer(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        else:
            self.attn = AutoCorrelationAttention(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Decomposition ---------------------------------------------------
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)

        # --- Convolutional feed-forward (1×1 conv pair) ----------------------
        hidden_dim = int(dim * mlp_ratio)
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.act = act_layer()
        self.dropout = nn.Dropout(proj_drop)

        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Final norm for stability (optional – comment out if undesired)
        # self.norm_out = norm_layer(dim)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape [B, L, C].
        attn_mask : Tensor, optional
            Broadcastable attention mask.

        Returns
        -------
        Tensor
            Same shape as input.
        """
        # --- 1. Attention ---------------------------------------------------
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))

        # --- 2. First decomposition ----------------------------------------
        x, _ = self.decomp1(x)                           # keep seasonal part

        # --- 3. Conv-MLP ----------------------------------------------------
        y = x.transpose(1, 2)                            # [B, C, L]
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)                            # back to [B, L, C]

        x = x + self.drop_path2(self.ls2(y))

        # --- 4. Second decomposition ---------------------------------------
        x, _ = self.decomp2(x)

        # return self.norm_out(x)                         # if extra norm wanted
        return x

class AutoformerDecodeBlock(nn.Module):
    """
    Autoformer decoder layer with progressive decomposition.
    Matches the TransformerBlock-style API (`norm1/self_attn/ls1/drop_path1 …`).
    """

    def __init__(
        self,
        *args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        moving_avg: int = 25,
        qkv_bias: bool = False,
        factor: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = LayerNorm,
        self_attn_layer: Optional[Type[nn.Module]] = None,
        cross_attn_layer: Optional[Type[nn.Module]] = None,
        **kwargs
    ) -> None:
        super(AutoformerDecodeBlock, self).__init__(*args, **kwargs)

        # --- Self Attention branch ---
        self.norm1 = norm_layer(dim)

        if self_attn_layer is not None:
            self.self_attn = self_attn_layer(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        else:
            self.self_attn = AutoCorrelationAttention(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- First Decomposition ---
        self.decomp1 = SeriesDecomp(moving_avg)

        # --- Cross Attention branch ---
        self.norm2 = norm_layer(dim)
        # Use AutoCorrelationAttention or a passed-in module for cross-attn
        if cross_attn_layer is not None:
            self.cross_attn = cross_attn_layer(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        else:
            self.cross_attn = AutoCorrelationAttention(
                dim,
                num_heads=num_heads,
                factor=factor,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
            )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Second Decomposition ---
        self.decomp2 = SeriesDecomp(moving_avg)

        # --- Conv-MLP branch ---
        hidden_dim = int(dim * mlp_ratio)
        self.conv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.act = act_layer()
        self.dropout = nn.Dropout(proj_drop)

        self.ls3 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Third Decomposition ---
        self.decomp3 = SeriesDecomp(moving_avg)

        # --- Projection for residual trend output ---
        self.proj = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )

    def forward(
            self,
            x: torch.Tensor,
            cross: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            cross_attn_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            [B, L, C]  (decoder input)
        cross : Tensor
            [B, S, C]  (encoder output)
        attn_mask : Tensor, optional
            Mask for self-attention.
        cross_attn_mask : Tensor, optional
            Mask for cross-attention.

        Returns
        -------
        x : Tensor
            [B, L, C]  (decoder output)
        residual_trend : Tensor
            [B, L, C]  (for final forecast fusion)
        """
        # 1. Self-attention
        x = x + self.drop_path1(self.ls1(self.self_attn(self.norm1(x), attn_mask)))
        # 2. First decomposition
        x, trend1 = self.decomp1(x)
        # 3. Cross-attention
        x = x + self.drop_path2(self.ls2(self.cross_attn(self.norm2(x),
                                                         cross,
                                                         cross,
                                                         cross_attn_mask)))
        # 4. Second decomposition
        x, trend2 = self.decomp2(x)
        # 5. Conv-MLP
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        x = x + self.drop_path3(self.ls3(y))
        # 6. Third decomposition
        x, trend3 = self.decomp3(x)
        # 7. Aggregate trends and project
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.proj(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend

if __name__ == '__main__':
    device = torch.device("cpu")

    # Example usage
    batch_size = 2
    seq_len = 10
    dim = 64
    num_heads = 4

    batch = torch.randn(batch_size, seq_len, dim, device=device).to(device)

    autoformer_encode_block = AutoformerEncodeBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        moving_avg=25,
        qkv_bias=True,
        factor=1,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm
    ).to(device)
    print(autoformer_encode_block)

    output = autoformer_encode_block(batch)

    print("Encoder output shape:", output.shape)  # Should be [batch_size, seq_len, dim]

    autoformer_decode_block = AutoformerDecodeBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        moving_avg=25,
        qkv_bias=True,
        factor=1,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm
    ).to(device)

    cross = torch.randn(batch_size, seq_len, dim, device=device).to(device)  # Simulated encoder output
    batch = torch.randn(batch_size, seq_len, dim, device=device).to(device)  # Decoder input
    output, residual_trend = autoformer_decode_block(batch, cross)
    print("Decoder output shape:", output.shape)  # Should be [batch_size, seq_len, dim]
    print("Residual trend shape:", residual_trend.shape)  # Should be [batch_size, seq_len, dim]
