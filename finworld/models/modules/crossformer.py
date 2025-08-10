import torch
import math
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Optional, Type, Final
from timm.layers import Mlp, LayerNorm, DropPath

from finworld.models.modules.attention import Attention

class TwoStageAttention(nn.Module):
    """
    Two‑Stage Attention (TSA) layer for time‑series shaped (B, L, N, C).

    Stage‑1 (time attention) uses the same `Attention` module you provided.
    Stage‑2 (cross‑dim attention) is realised with two standard
    `nn.MultiheadAttention` calls (router ←→ stocks).

    Args:
        seg_num (int): Number of temporal segments **L**.
        factor (int): Number of learnable router tokens per segment.
        dim (int): Token embedding dimension **E**.
        num_heads (int): Attention heads.
        qkv_bias, qk_norm, scale_norm, proj_bias, attn_drop, proj_drop,
        norm_layer: Keep exactly the same semantics as the original `Attention`.
        mlp_ratio (float): Expansion factor of the feed‑forward network.
        dropout (float): Dropout used in residual branches.
    Shapes:
        Input  : (B, L, N, C)
        Output : (B, L, N, C)
    """

    fused_attn: Final[bool]  # kept only to stay parallel with `Attention`

    def __init__(
        self,
        seg_num: int,
        factor: int,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qk_norm or scale_norm:
            assert norm_layer is not None, "`norm_layer` is required when qk_norm/scale_norm is True"

        # ---------- Stage‑1 : intra‑stock (time) attention ----------
        self.time_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        # ---------- Stage‑2 : router‑mediated cross‑stock attention ----------
        self.dim_sender = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )   # Q = router, K/V = stock

        self.dim_receiver = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )   # Q = stock,  K/V = router

        # Learnable router tokens: (L, factor, C)
        self.router = nn.Parameter(torch.randn(seg_num, factor, dim))

        # ---------- Feed‑forward network ----------
        hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # ---------- Normalisation ----------
        self.norm1 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)
        self.norm2 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)
        self.norm3 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)
        self.norm4 = norm_layer(dim) if norm_layer else nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # keep attribute for parity with `Attention`
        self.fused_attn = getattr(self.time_attn, "fused_attn", False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,   # kept for API symmetry
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape (B, L, N, C)
            attn_mask (Optional[Tensor]): Only passed to Stage‑1 for now.
        Returns:
            Tensor: Output of shape (B, L, N, C)
        """
        B, L, N, C = x.shape
        assert L == self.router.shape[0], \
            f"seg_num mismatch: router has {self.router.shape[0]}, input has {L}"

        # -------- Stage‑1 : time attention (per stock) --------
        time_in  = rearrange(x, "b l n c -> (b n) l c")           # (B*N, L, C)
        time_enc = self.time_attn(time_in, attn_mask)             # (B*N, L, C)
        z = self.norm1(time_in + self.drop(time_enc))
        z = self.norm2(z + self.drop(self.mlp1(z)))               # (B*N, L, C)

        # -------- Stage‑2 : cross‑stock attention via router --------
        dim_send = rearrange(z, "(b n) l c -> (b l) n c", b=B)    # (B*L, N, C)
        routers  = repeat(self.router, "l r c -> (b l) r c", b=B) # (B*L, factor, C)

        # Router gathers messages from stocks
        buf, _ = self.dim_sender(routers, dim_send, dim_send)     # (B*L, factor, C)
        # Stocks receive fused messages
        dim_rec, _ = self.dim_receiver(dim_send, buf, buf)        # (B*L, N, C)

        y = self.norm3(dim_send + self.drop(dim_rec))
        y = self.norm4(y + self.drop(self.mlp2(y)))               # (B*L, N, C)

        out = rearrange(y, "(b l) n e -> b l n e", b=B, l=L)
        return out

# =========================================
# 1) Seg‑Merging for (B, L, N, C)
# =========================================
class SegMerging(nn.Module):
    """
    Merge `win_size` neighbouring temporal segments along **L** while
    leaving (N, C) untouched.

    Input : (B, L, N, C)
    Output: (B, ceil(L / win_size), N, C)
    """
    def __init__(self, dim: int, win_size: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.norm   = norm_layer(win_size * dim)
        self.linear = nn.Linear(win_size * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, C = x.shape

        # --- pad so that L is divisible by win_size ---
        pad_len = (-L) % self.win_size
        if pad_len:
            pad_seg = x[:, -pad_len:, ...]            # repeat last seg
            x = torch.cat([x, pad_seg], dim=1)
            L = x.size(1)

        # --- group & concat along channel dimension ---
        # reshape → (B, L//w, w, N, C) → (B, L//w, N, w*C)
        w = self.win_size
        x = x.view(B, L // w, w, N, C).permute(0, 1, 3, 2, 4).reshape(B, L // w, N, w * C)

        # --- fuse & project back to C ---
        x = self.norm(x)
        x = self.linear(x)                            # (B, L//w, N, C)
        return x


# =========================================
# 2) ScaleBlock  (SegMerge  +  TwoStageAttention + MLP)
# =========================================
class CrossformerEncodeBlock(nn.Module):
    """
    Hierarchical block that (optionally) downsamples time segments and then
    applies Two‑Stage Attention followed by an MLP, mimicking TransformerBlock
    but operating on (B, L, N, C) throughout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seg_num: int,
        win_size: int = 1,
        factor: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = None,
        attn_layer: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()

        # ---- optional temporal down‑sampling ----
        self.merge = SegMerging(dim, win_size, norm_layer) if win_size > 1 else None
        merged_L = math.ceil(seg_num / win_size)

        # ---- Two‑Stage Attention ----
        self.norm1 = norm_layer(dim)

        if attn_layer is not None:  # custom attention layer
            self.attn = attn_layer(
                seg_num=merged_L,
                factor=factor,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_norm=scale_attn_norm,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )
        else:  # default TwoStageAttention
            self.attn  = TwoStageAttention(
                seg_num=merged_L,
                factor=factor,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_norm=scale_attn_norm,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )

        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.dp1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ---- MLP ----
        hidden_dim = int(dim * mlp_ratio)
        if mlp_layer is None:          # fallback
            mlp_layer = lambda in_f, hid_f: nn.Sequential(
                nn.Linear(in_f, hid_f),
                act_layer(),
                nn.Dropout(proj_drop),
                nn.Linear(hid_f, in_f),
                nn.Dropout(proj_drop),
            )
        self.norm2 = norm_layer(dim)
        self.mlp   = mlp_layer(dim, hidden_dim)
        self.ls2   = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.dp2   = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, N, C)
        """
        # 1) SegMerging (optional)
        if self.merge is not None:
            x = self.merge(x)   # (B, L', N, C)

        # 2) Two‑Stage Attention
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))

        # 3) Feed‑forward
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CrossformerDecodeBlock(nn.Module):
    """
    Crossformer decoder layer (Transformer‑style).

    * norm1 → Two‑Stage self‑attention
    * norm2 → Multi‑Head cross‑attention (decoder ↔ encoder)
    * Conv‑MLP branch
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        seg_num: int,
        win_size: int = 1,
        factor: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        self_attn_layer: Optional[Type[nn.Module]] = None,  # custom self-attention layer
        cross_attn_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # --- Self‑Attention branch ---
        self.norm1 = norm_layer(dim)

        if self_attn_layer is not None:  # custom attention layer
            self.self_attn = self_attn_layer(
                seg_num=seg_num,
                factor=factor,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
        else:  # default TwoStageAttention
            self.self_attn = TwoStageAttention(
                seg_num=seg_num,
                factor=factor,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer
            )

        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.dp1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Cross‑Attention branch ---
        self.norm2 = norm_layer(dim)

        if cross_attn_layer is not None:  # custom cross-attention layer
            self.cross_attn = cross_attn_layer(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_drop,
                batch_first=True,
                bias=qkv_bias,
            )
        else:  # default MultiheadAttention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_drop,
                batch_first=True,
                bias=qkv_bias,
            )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Conv‑MLP branch ---
        hidden_dim = int(dim * mlp_ratio)
        self.conv1 = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv1d(hidden_dim, dim, 1, bias=False)
        self.act = act_layer()
        self.dropout = nn.Dropout(proj_drop)
        self.ls3 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.dp3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,            # decoder queries   [B, L, N, C]
        cross: torch.Tensor,        # encoder outputs   [B, L_enc, N, C]
        attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, N, C = x.shape

        # 1. Two‑Stage self‑attention
        x = x + self.dp1(self.ls1(self.self_attn(self.norm1(x), attn_mask)))

        # 2. Cross‑attention (reshape to [B, L*N, C])
        q = rearrange(x, 'b l n c -> b (l n) c')  # [B, L*N, C]
        kv = rearrange(cross, 'b l n c -> b (l n) c')  # [B, L_enc*N, C]
        attn_out, _ = self.cross_attn(q, kv, kv, attn_mask=cross_attn_mask, need_weights=False)

        # 3. Residual and Conv‑MLP
        q = q + self.dp2(self.ls2(attn_out))
        q_norm = self.norm2(q)

        y = q_norm.transpose(1, 2)        # [B, C, L*N]
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)             # [B, L*N, C]

        q = q_norm + self.dp3(self.ls3(y))
        dec_output = q                    # [B, L*N, C]

        # 4. Restore original shape
        dec_output = rearrange(dec_output, 'b (l n) c -> b l n c', b=B, l=L)

        return dec_output

if __name__ == '__main__':
    # Example usage
    batch_size = 2
    num_stocks = 5
    encode_seg_num = 16
    decode_seg_num = 8
    embedding_dim = 128
    factor = 1
    num_heads = 4

    x = torch.randn(batch_size, encode_seg_num, num_stocks, embedding_dim)

    block = CrossformerEncodeBlock(
        dim=embedding_dim,
        num_heads=num_heads,
        seg_num=encode_seg_num,
        win_size=1,
        factor=factor,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        scale_attn_norm=False,
        scale_mlp_norm=False,
        proj_bias=True,
        proj_drop=0.1,
        attn_drop=0.1,
        init_values=None,
        drop_path=0.1,
    )
    output = block(x)
    print(output.shape)

    x = torch.randn(batch_size, decode_seg_num, num_stocks, embedding_dim)
    block = CrossformerDecodeBlock(
        dim=embedding_dim,
        num_heads=num_heads,
        seg_num=decode_seg_num,
        win_size=1,
        factor=factor,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=0.1,
        attn_drop=0.1,
        init_values=None,
        drop_path=0.1,
    )
    cross = torch.randn(batch_size, encode_seg_num, num_stocks, embedding_dim)  # Simulated encoder output
    dec_output = block(x, cross)
    print(dec_output.shape)  # Should be (batch_size, seg_num, num_stocks, embedding_dim)