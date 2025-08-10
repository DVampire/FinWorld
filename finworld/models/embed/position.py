import torch
import torch.nn as nn
from typing import Optional
from timm.models.layers import to_2tuple
import torch.nn.functional as F
import math

from finworld.registry import EMBED

@EMBED.register_module(force=True)
class AbsPosition1DEmbed(nn.Module):
    """
    Learnable absolute 1‑D position embedding.

    ✔  Supports an arbitrary number of **prefix tokens** (e.g. [CLS], [DISTILL]).
    ✔  Automatically **interpolates** when the input sequence is longer than
       the table generated at init time, so you can fine‑tune / perform inference
       on sequences of different length without touching the checkpoint.
    """

    def __init__(
        self,
        num_positions: int = 64,          # time steps (without prefix tokens)
        embed_dim: int = 128,             # hidden size (must match model)
        num_prefix: int = 0        # e.g. 1 for a single [CLS] token
    ):
        super().__init__()

        self.num_prefix = num_prefix
        self.max_len = num_positions + num_prefix

        # ------------------------------------------------------------------ #
        # Learnable table:  shape (1, max_len, embed_dim)
        #   Prefix rows are placed *first* so they align with tokens that are
        #   normally prepended to the input sequence.
        # ------------------------------------------------------------------ #
        self.pos_table = nn.Parameter(torch.zeros(1, self.max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_table, std=0.02)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
            self,
            x: torch.Tensor,
            has_prefix: bool = False,  # does `x` already contain prefix tokens?
    ) -> torch.Tensor:
        """
        Args
        ----
        x : (B, L, D)    token embeddings.
                         If `has_prefix == True`, L already includes `num_prefix` rows.
                         Otherwise L is *pure* sequence length without prefix tokens.

        Returns
        -------
        Tensor of the **same** shape as `x`, but shifted by its absolute position
        embedding.  No extra tokens are inserted inside this method.
        """
        B, L, _ = x.shape

        # ------------------------------------------------------------------ #
        # 1. We need at least `start_row + L` rows in the position table.
        #    `start_row = 0`  when the sequence already contains prefix tokens
        #    `start_row = num_prefix` when it doesn't.
        # ------------------------------------------------------------------ #
        start_row = 0 if has_prefix else self.num_prefix
        total_rows = start_row + L

        # Fetch / interpolate to the required number of rows
        pos_full = self._get_pos_embed(total_rows).to(x.dtype).to(x.device)  # (1, total_rows, D)

        # ------------------------------------------------------------------ #
        # 2. Slice out exactly the rows that correspond to the input tokens
        # ------------------------------------------------------------------ #
        pos_slice = pos_full[:, start_row: start_row + L]  # (1, L, D)

        # 3. Add & return (shape is identical to `x`)
        return x + pos_slice

    # ---------------------------------------------------------------------- #
    # Internal helper: trim or linearly interpolate to the desired length
    # ---------------------------------------------------------------------- #
    def _get_pos_embed(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.pos_table.shape[1]:
            # Shorter or equal length → direct slice
            return self.pos_table[:, :seq_len]

        # Longer sequence → 1‑D linear interpolation
        pe = self.pos_table.transpose(1, 2)                     # (1, D, L_old)
        pe = F.interpolate(
            pe, size=seq_len, mode="linear", align_corners=False
        )
        return pe.transpose(1, 2)                               # (1, L_new, D)

@EMBED.register_module(force=True)
class AbsPosition2DEmbed(nn.Module):
    """
    Factorised absolute position embedding for a flattened (time × space) grid.

    Input  : (B, L, C) where L = num_prefix + T * N
    Output : same shape (B, L, C)
    """

    def __init__(
        self,
        num_time: int = 64,        # initial T  (>=1)
        num_space: int = 32,       # initial N  (>=1)
        embed_dim: int = 128,      # hidden size
        num_prefix: int = 0        # e.g. 1 for a single [CLS]
    ):
        super().__init__()
        assert num_time > 0 and num_space > 0
        assert embed_dim > 0

        self.num_prefix = num_prefix

        # learnable 1‑D tables
        self.time_table  = nn.Parameter(torch.zeros(1, num_time,  embed_dim))
        self.space_table = nn.Parameter(torch.zeros(1, num_space, embed_dim))
        nn.init.trunc_normal_(self.time_table,  std=.02)
        nn.init.trunc_normal_(self.space_table, std=.02)

        # learnable prefix rows (optional)
        if num_prefix:
            self.prefix_table = nn.Parameter(torch.zeros(1, num_prefix, embed_dim))
            nn.init.trunc_normal_(self.prefix_table, std=.02)
        else:
            self.register_parameter("prefix_table", None)

    # -------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,             # (B, L, C)  flattened seq
        time_len: Optional[int] = None,
        *,
        has_prefix: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, L, C) flattened token embeddings.
        time_len  : int or None.  If None, we auto‑infer T from L and num_prefix.
        has_prefix: whether `x` already contains `num_prefix` prefix tokens.

        Returns
        -------
        Pos‑embedded tensor with the **same shape** as `x`.
        """
        B, L, C = x.shape
        if C != self.time_table.shape[-1]:
            raise ValueError("embed_dim mismatch")

        # 1. split prefix / body ------------------------------------------------
        if self.num_prefix:
            if has_prefix:
                if L < self.num_prefix:
                    raise ValueError("Sequence shorter than `num_prefix`.")
                x_prefix = x[:, : self.num_prefix]         # (B, P, C)
                x_body   = x[:, self.num_prefix :]         # (B, *, C)
            else:
                x_prefix = None
                x_body   = x                               # (B, *, C)
        else:
            if has_prefix:
                raise ValueError("Model built with num_prefix=0 but has_prefix=True")
            x_prefix = None
            x_body   = x

        body_len = x_body.shape[1]

        # 2. infer or validate grid size ----------------------------------------
        if time_len is None:
            if body_len == 0:
                raise ValueError("Cannot infer grid shape from empty body.")
            # try every divisor up to sqrt(body_len)
            time_len = self._infer_time(body_len)
            if time_len is None:
                raise ValueError(
                    "`time_len` not provided and cannot be inferred; "
                    "please pass an explicit value."
                )
        if body_len % time_len != 0:
            raise ValueError("`time_len` does not divide sequence body length.")
        space_len = body_len // time_len  # N

        # 3. fetch / interpolate tables -----------------------------------------
        e_t = self._resize_1d(self.time_table,  time_len, x.dtype, x.device)  # (1,T,C)
        e_n = self._resize_1d(self.space_table, space_len, x.dtype, x.device) # (1,N,C)

        # flatten to match ordering: time‑major (t0n0, t0n1, ..., t1n0, ...)
        pos_time  = e_t.repeat_interleave(space_len, dim=1)   # (1,T*N,C)
        pos_space = e_n.repeat(1, time_len, 1)                # (1,T*N,C)
        pos_body  = pos_time + pos_space                      # (1,body_len,C)

        # 4. prefix rows ---------------------------------------------------------
        if self.num_prefix:
            pos_prefix = self.prefix_table  # (1, P, C)

            if has_prefix:  # has prefix tokens
                # insert prefix rows at the front
                pos_full = torch.cat([pos_prefix, pos_body], dim=1)  # (1, P+T*N, C)

            else:  # <-- has_prefix=False
                # not inserting prefix rows, just use body positions
                pos_full = pos_body  # (1, T*N, C)
        else:
            pos_full = pos_body

        # 5. add & return --------------------------------------------------------
        return x + pos_full.to(x.dtype)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _resize_1d(
        table: torch.Tensor,
        tgt_len: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """
        Slice or linearly interpolate a learnable 1‑D table to length `tgt_len`.
        Always returns (1, tgt_len, C) on the requested device/dtype.
        """
        cur_len = table.shape[1]
        if tgt_len <= cur_len:
            return table[:, :tgt_len].to(device=device, dtype=dtype)

        pe = table.to(device=device, dtype=dtype).transpose(1, 2)      # (1,C,L0)
        pe = F.interpolate(pe, size=tgt_len, mode="linear", align_corners=False)
        return pe.transpose(1, 2)                                      # (1,L_new,C)

    @staticmethod
    def _infer_time(body_len: int) -> Optional[int]:
        """
        Heuristic: pick the **largest divisor <= sqrt(body_len)** as T.
        Works well when grid is approximately square; otherwise user should
        pass `time_len` explicitly.
        """
        root = int(math.sqrt(body_len))
        for t in range(root, 0, -1):
            if body_len % t == 0:
                return t
        return None


@EMBED.register_module(force=True)
class SinCosPosition1DEmbed(nn.Module):
    """
    Non‑learnable 1‑D sine–cosine position embedding (ViT/MAE style).

    ✔  Supports an arbitrary number of **prefix tokens** (e.g. [CLS], [DISTILL]).
    ✔  Automatically **extends** itself when the sequence is longer than the
       table created at init time (no interpolation error, we just regenerate).

    Compared with `AbsPosition1DEmbed`, this variant has **zero parameters**,
    so it is smaller and naturally extrapolates to unseen sequence lengths.
    """

    def __init__(
        self,
        num_positions: int = 64,     # time steps (WITHOUT prefix tokens)
        embed_dim: int = 128,        # hidden size (must match the backbone)
        num_prefix: int = 0          # 1 → single [CLS] token at the front
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "`embed_dim` must be an even number"

        self.num_prefix = num_prefix
        self.max_len = num_positions + num_prefix
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------ #
        # Pre‑generate a table up to `max_len`; store as a buffer so it moves
        # with `.to()/.cuda()` but is NOT a learnable parameter.
        # ------------------------------------------------------------------ #
        init_table = self._build_sincos(self.max_len, embed_dim)  # (1,L,D)
        self.register_buffer("pos_table", init_table, persistent=False)

    # ---------------------------------------------------------------------- #
    # Public forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        has_prefix: bool = False     # does `x` already contain prefix tokens?
    ) -> torch.Tensor:
        """
        Args
        ----
        x : (B, L, D)  — token embeddings.
            If `has_prefix == True`, L already includes `num_prefix` tokens.
            Otherwise L is *pure* sequence length without prefix tokens.

        Returns
        -------
        Tensor of identical shape to `x`, offset by its sine–cosine position
        embedding.  No extra tokens are inserted here.
        """
        B, L, _ = x.shape

        # Which row in the table corresponds to the first token in `x`?
        start_row = 0 if has_prefix else self.num_prefix
        total_rows = start_row + L

        # Fetch or (re‑)generate enough rows
        pos_full = self._get_pos_embed(total_rows, dtype=x.dtype,
                                       device=x.device)          # (1,total,D)
        pos_slice = pos_full[:, start_row : start_row + L]        # (1,L,D)

        return x + pos_slice

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def _get_pos_embed(
        self,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """
        Return a (1, seq_len, D) position table on the correct device/dtype.
        If `seq_len` exceeds the pre‑generated table, build a new one on‑the‑fly.
        """
        if seq_len <= self.pos_table.shape[1]:
            return self.pos_table[:, :seq_len].to(dtype=dtype, device=device)

        # Too long → build from scratch and cache for possible future calls
        new_table = self._build_sincos(seq_len, self.embed_dim).to(device).to(dtype)
        # (Optional) cache the enlarged table to avoid recomputation next time
        self.pos_table = new_table.to(self.pos_table.device, self.pos_table.dtype)
        return new_table

    @staticmethod
    def _build_sincos(length: int, dim: int) -> torch.Tensor:
        """
        Create a (1, length, dim) sine–cosine position matrix.
        """
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)       # (L,1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / dim)
        )                                                                       # (D/2,)

        pe = torch.zeros(length, dim, dtype=torch.float32)                      # (L,D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)                                                  # (1,L,D)

@EMBED.register_module(force=True)
class SinCosPosition2DEmbed(nn.Module):
    """
    Factorised 2‑D **sine–cosine** absolute position embedding for a flattened
    (time × space) grid.  No learnable parameters ─ suitable for zero‑shot
    length extrapolation.

      • Input  : (B, L, C)  with  L = num_prefix + T * N
      • Output : same shape (B, L, C)

    Each token at (t, n) receives

        sincos_time[t]  +  sincos_space[n]

    Prefix tokens (if any) are assigned **all‑zero** vectors.
    """

    def __init__(
        self,
        num_time: int = 64,     # default T used to build the initial table
        num_space: int = 32,    # default N
        embed_dim: int = 128,   # model width  (must be even)
        num_prefix: int = 0     # e.g. 1 for [CLS]
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "`embed_dim` must be even for sin‑cos encoding."
        self.num_prefix = num_prefix
        self.embed_dim = embed_dim

        # pre‑generate sin‑cos tables up to (num_time / num_space)
        t_table = self._build_sincos(num_time, embed_dim)    # (1,T,C)
        n_table = self._build_sincos(num_space, embed_dim)   # (1,N,C)

        self.register_buffer("time_table",  t_table,  persistent=False)
        self.register_buffer("space_table", n_table,  persistent=False)

        if num_prefix:
            prefix_zeros = torch.zeros(1, num_prefix, embed_dim)
            self.register_buffer("prefix_table", prefix_zeros, persistent=False)
        else:
            self.register_buffer("prefix_table", None, persistent=False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,              # (B, L, C)
        time_len: Optional[int] = None,
        *,
        has_prefix: bool = False
    ) -> torch.Tensor:
        B, L, C = x.shape
        if C != self.embed_dim:
            raise ValueError("embed_dim mismatch.")

        # -------- split prefix / body ----------------------------------- #
        if self.num_prefix:
            if has_prefix:
                x_prefix = x[:, : self.num_prefix]
                x_body   = x[:, self.num_prefix :]
            else:
                x_prefix = None
                x_body   = x
        else:
            if has_prefix:
                raise ValueError("Model built with num_prefix=0 but got has_prefix=True")
            x_prefix = None
            x_body   = x

        body_len = x_body.shape[1]

        # -------- infer / check grid shape ------------------------------ #
        if time_len is None:
            time_len = self._infer_time(body_len)
            if time_len is None:
                raise ValueError("Cannot infer `time_len`; please pass it explicitly.")
        if body_len % time_len != 0:
            raise ValueError("`time_len` does not divide sequence body length.")
        space_len = body_len // time_len

        # -------- fetch (or build) sin‑cos tables ------------------------ #
        e_t = self._get_1d(self.time_table,  time_len,  x.device, x.dtype)  # (1,T,C)
        e_n = self._get_1d(self.space_table, space_len, x.device, x.dtype)  # (1,N,C)

        pos_time  = e_t.repeat_interleave(space_len, dim=1)  # (1,T*N,C)
        pos_space = e_n.repeat(1, time_len, 1)               # (1,T*N,C)
        pos_body  = pos_time + pos_space                     # (1,body_len,C)

        # -------- handle prefix ----------------------------------------- #
        if self.num_prefix:
            if has_prefix:
                pos_full = torch.cat([self.prefix_table, pos_body], dim=1)  # (1,L,C)
            else:
                pos_full = pos_body                                         # (1,L,C)
        else:
            pos_full = pos_body

        return x + pos_full.to(dtype=x.dtype)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _get_1d(
        self,
        table: torch.Tensor,
        tgt_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return a (1, tgt_len, C) sin‑cos table on the requested device/dtype.
        Generates a new table if tgt_len exceeds the cached one.
        """
        cur_len = table.shape[1]
        if tgt_len <= cur_len:
            return table[:, :tgt_len].to(device=device, dtype=dtype)

        # need a larger table → rebuild and (optionally) cache
        new_table = self._build_sincos(tgt_len, self.embed_dim).to(device).to(dtype)
        # cache for future calls
        return new_table

    @staticmethod
    def _build_sincos(length: int, dim: int) -> torch.Tensor:
        """
        Create a (1, length, dim) 1‑D sine–cosine position matrix.
        """
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / dim)
        )                                                                  # (D/2,)
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)                                             # (1,L,D)

    @staticmethod
    def _infer_time(body_len: int) -> Optional[int]:
        """largest divisor ≤ √body_len"""
        root = int(math.sqrt(body_len))
        for t in range(root, 0, -1):
            if body_len % t == 0:
                return t
        return None
    


if __name__ == '__main__':
    device = torch.device("cpu")

    model = AbsPosition1DEmbed(num_positions=64, embed_dim=128, num_prefix=1)
    batch = torch.randn(2, 64 + 1, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, has_prefix=True)
    print(output.shape)  # Should be (2, 64 + 1, 128)
    batch = torch.randn(2, 64, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, has_prefix=False)
    print(output.shape)  # Should be (2, 64, 128) without prefix tokens

    model = SinCosPosition1DEmbed(num_positions=64, embed_dim=128, num_prefix=1)
    batch = torch.randn(2, 64 + 1, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, has_prefix=True)
    print(output.shape)  # Should be (2, 64 + 1, 128)
    batch = torch.randn(2, 64, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, has_prefix=False)
    print(output.shape)  # Should be (2, 64, 128) without prefix tokens

    model = AbsPosition2DEmbed(num_time=64, num_space=32, embed_dim=128, num_prefix=1)
    batch = torch.randn(2, 64 * 32 + 1, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, time_len=64, has_prefix=True)
    print(output.shape)  # Should be (2, 64 * 32 + 1, 128)
    batch = torch.randn(2, 64 * 32, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, time_len=64, has_prefix=False)
    print(output.shape)  # Should be (2, 64 * 32, 128) without prefix tokens

    model = SinCosPosition2DEmbed(num_time=64, num_space=32, embed_dim=128, num_prefix=1)
    batch = torch.randn(2, 64 * 32 + 1, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, time_len=64, has_prefix=True)
    print(output.shape)  # Should be (2, 64 * 32 + 1, 128)
    batch = torch.randn(2, 64 * 32, 128).to(device)  # (batch_size, seq_len, embed_dim)
    output = model(batch, time_len=64, has_prefix=False)
    print(output.shape)  # Should be (2, 64 * 32, 128) without prefix tokens