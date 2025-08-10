import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from typing import Optional, Type
from einops import rearrange, reduce, repeat
import math, random
from scipy.fftpack import next_fast_len


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)

def conv1d_fft(f, g, dim=-1):
    N = f.size(dim)
    M = g.size(dim)

    fast_len = next_fast_len(N + M - 1)

    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)

    F_fg = F_f * F_g.conj()
    out = fft.irfft(F_fg, fast_len, dim=dim)
    out = out.roll((-1,), dims=(dim,))
    idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
    out = out.index_select(dim, idx)

    return out

class ExponentialSmoothing(nn.Module):
    """Exponential smoothing kernel used in *Etsformer*–style architectures.

    The layer maintains a learnable smoothing parameter *α* (one per head) that
    defines an exponentially decaying 1‑D convolutional kernel. At each
    timestep *t*, the kernel weight is *(1–α)·αᵗ*.  A separate vector
    ``v0`` is blended into the output to capture the long‑term initial value.

    Parameters
    ----------
    dim : int
        Embedding dimension *per head* (i.e. channel count).
    num_heads : int
        Number of attention heads.  The smoothing weight :math:`\alpha` is
        learned independently for every head.
    dropout : float, optional
        Drop‑out probability applied to the *value* tensor(s) before the FFT
        convolution.  Defaults to ``0.1``.
    aux : bool, optional
        If *True*, an auxiliary value stream can be passed in during the
        forward call and will be combined using the same exponential kernel.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        aux: bool = False,
    ) -> None:
        super().__init__()

        # Learnable smoothing parameter α for each head, constrained to (0,1)
        self._smoothing_weight = nn.Parameter(torch.randn(num_heads, 1))

        # Trainable initial value v₀ blended in with weight αᵗ
        self.v0 = nn.Parameter(torch.randn(1, 1, num_heads, dim))

        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        values: torch.Tensor,
        aux_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the exponential smoothing kernel.

        Parameters
        ----------
        values : Tensor
            The primary value stream of shape ``[B, T, H, D]``.
        aux_values : Tensor, optional
            Optional auxiliary value stream *with the same shape* used by the
            original Etsformer implementation to model residual trend.

        Returns
        -------
        Tensor
            Smoothed values with identical shape to *values*.
        """
        b, t, h, d = values.shape  # batch, time, heads, dim

        # Construct the convolution kernel (length=T)
        init_weight, kernel = self._get_exponential_weight(t)

        # FFT‑based circular convolution along the time axis
        output = conv1d_fft(self.dropout(values), kernel, dim=1)
        # Blend in the initial value component αᵗ · v₀
        output = init_weight * self.v0 + output

        # Optional auxiliary stream – uses a re‑weighted kernel so that both
        # streams sum to the same total contribution.
        if aux_values is not None:
            #  kernel / (1–α) * α  =  α · αᵗ = αᵗ⁺¹
            aux_kernel = kernel / (1 - self.weight) * self.weight
            aux_out = conv1d_fft(self.aux_dropout(aux_values), aux_kernel, dim=1)
            output = output + aux_out

        return output

    # ------------------------------------------------------------------ #
    def _get_exponential_weight(
        self, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exponential kernel weights for a given sequence length.

        Returns
        -------
        (init_weight, kernel) : tuple[Tensor, Tensor]
            * ``init_weight`` – αᵗ for *t*∈[1,…,T]; shape ``[1, T, H, 1]``
            * ``kernel``      – (1–α)·αᵗ for *t*∈[T–1,…,0]; same shape
        """
        # Time indices 0,1,…,T–1 on the device of α
        powers = torch.arange(seq_len, dtype=torch.float32, device=self.weight.device)

        # (1–α)·αᵗ  for t = T–1, T–2, …, 0  (reverse order)
        kernel = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))

        # αᵗ  for t = 1,2,…,T
        init_weight = self.weight ** (powers + 1)

        # Reshape to [1, T, H, 1] to broadcast over batch and channel dims
        return (
            rearrange(init_weight, "h t -> 1 t h 1"),
            rearrange(kernel, "h t -> 1 t h 1"),
        )

    # ------------------------------------------------------------------ #
    @property
    def weight(self) -> torch.Tensor:
        """The smoothing parameter α constrained to the open interval (0,1)."""
        return torch.sigmoid(self._smoothing_weight)


class FeedForward(nn.Module):
    """
    Position‑wise feed‑forward network used inside Transformer blocks.

    Architecture
    ------------
    Linear(dim → latent_dim) → Activation → Dropout →
    Linear(latent_dim → dim) → Dropout

    Parameters
    ----------
    dim : int
        Channel dimension of the incoming sequence (a.k.a. model width).
    latent_dim : int
        Width of the inner (expansion) layer.
    dropout : float, default 0.1
        Drop probability applied after each linear projection.
    activation : str | Callable, default "sigmoid"
        Non‑linear function; either the name of a function in
        `torch.nn.functional` or a custom callable.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        # Two linear projections (bias disabled to mirror the original logic)
        self.fc1 = nn.Linear(dim, latent_dim, bias=False)
        self.fc2 = nn.Linear(latent_dim, dim, bias=False)

        # Independent dropout masks
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # Resolve the activation function once for efficiency
        self.act = act_layer()

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  – shape **[B, L, dim]**

        Returns
        -------
        Tensor    – shape **[B, L, dim]**
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GrowthLayer(nn.Module):
    """
    Growth layer from the Etsformer family.

    Input / output shape
    --------------------
    Tensor **[B, T, dim]**
      B – batch size
      T – sequence length
      dim – model width  (must be divisible by `num_heads`)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)

        if self.head_dim * self.num_heads != self.dim:
            raise ValueError("`dim` must be divisible by `num_heads`.")

        # Learnable offset z₀ added in front of the sequence (per head)
        self.z0 = nn.Parameter(torch.randn(self.num_heads, self.head_dim))

        # In‑ and out‑projections (no bias, mirrors original implementation)
        self.in_proj  = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        # Exponential smoothing on head‑wise, per‑step deltas
        self.es = ExponentialSmoothing(
            dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=dropout,
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  – shape **[B, T, dim]**

        Returns
        -------
        Tensor     – shape **[B, T, dim]**
        """
        b, t, _ = x.shape

        # 1) Project to (head, head_dim) space
        v = rearrange(self.in_proj(x), 'b t (h d) -> b t h d', h=self.num_heads)

        # 2) Pre‑pend z₀ and take first differences  Δv_t = v_t − v_{t−1}
        z0 = repeat(self.z0, 'h d -> b 1 h d', b=b)      # [B, 1, H, D_h]
        delta = torch.cat([z0, v], dim=1)                # [B, T+1, H, D_h]
        delta = delta[:, 1:] - delta[:, :-1]             # [B, T,  H, D_h]

        # 3) Exponential smoothing of the deltas
        smoothed = self.es(delta)                        # [B, T, H, D_h]

        # 4) Re‑attach the initial v₀ term required by the paper
        smoothed = torch.cat(
            [repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), smoothed],
            dim=1,
        )                                                # [B, T+1, H, D_h]

        # 5) Merge heads and return to original channel count
        smoothed = rearrange(smoothed, 'b t h d -> b t (h d)')
        return self.out_proj(smoothed)

class FourierLayer(nn.Module):
    """
    Forecasting layer that extrapolates a sequence in the Fourier domain
    (cf. Etsformer variants).

    Parameters
    ----------
    dim : int
        Channel dimension of the input sequence.
    output_dim : int               <-- renamed from *pred_len*
        Number of future time‑steps to extrapolate.
    k : int | None, default None
        If given, keep only the *k* dominant frequency components.
        When `None`, use all available frequencies.
    low_freq : int, default 1
        Index of the lowest frequency to keep (0 == DC term).
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        k: int | None = None,
        low_freq: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.k = k
        self.low_freq = low_freq

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor – shape **[B, T, dim]**

        Returns
        -------
        Tensor  – shape **[B, T + output_dim, dim]**
        """
        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)                      # [B, ⌈T/2⌉+1, dim]

        # 1) Drop unwanted low‑frequency bins (optionally the Nyquist bin)
        if t % 2 == 0:                                  # even‑length FFT ⇒ Nyquist term present
            x_freq = x_freq[:, self.low_freq:-1]
            freqs   = fft.rfftfreq(t, device=x_freq.device)[self.low_freq:-1]
        else:                                           # odd length – no Nyquist
            x_freq = x_freq[:, self.low_freq:]
            freqs   = fft.rfftfreq(t, device=x_freq.device)[self.low_freq:]

        # 2) Optionally keep only the k largest‑magnitude frequencies
        x_freq, index_tuple = self._topk_freq(x_freq)

        # 3) Prepare frequency tensor for broadcasting during extrapolation
        freqs = repeat(freqs, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        freqs = rearrange(freqs[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        # 4) Reconstruct & extrapolate back to time domain
        return self._extrapolate(x_freq, freqs, t)

    # ------------------------------------------------------------------ #
    # Helper: extrapolate in time domain
    def _extrapolate(self, x_freq: torch.Tensor, freqs: torch.Tensor, t_in: int) -> torch.Tensor:
        """
        Given selected complex Fourier coefficients & their frequencies,
        reconstruct both the original and future samples.
        """
        # Mirror to create negative frequencies (ensures real output)
        x_full = torch.cat([x_freq, x_freq.conj()], dim=1)
        freqs  = torch.cat([freqs, -freqs], dim=1)

        # Time indices for original + extrapolated part
        t_vals = torch.arange(t_in + self.output_dim, dtype=torch.float32, device=x_freq.device)
        t_vals = rearrange(t_vals, 't -> () () t ()')

        # Polar form: amplitude & phase
        amp   = rearrange(x_full.abs() / t_in, 'b f d -> b f () d')
        phase = rearrange(x_full.angle(),       'b f d -> b f () d')

        # Cosine synthesis Σ_k A_k cos(2π f_k t + φ_k)
        x_time = amp * torch.cos(2 * math.pi * freqs * t_vals + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    # ------------------------------------------------------------------ #
    # Helper: keep top‑k frequencies by magnitude
    def _topk_freq(self, x_freq: torch.Tensor):
        """
        Pick the `k` frequency bins with the highest magnitude
        (per batch‑item and per channel), return them together with an index
        tuple for later gathering of the corresponding frequency values.
        """
        if self.k is None or self.k >= x_freq.size(1):
            # Use all frequencies
            b_idx, d_idx = torch.meshgrid(
                torch.arange(x_freq.size(0), device=x_freq.device),
                torch.arange(x_freq.size(2), device=x_freq.device),
                indexing='ij',
            )
            index_tuple = (b_idx.unsqueeze(1), torch.arange(x_freq.size(1), device=x_freq.device), d_idx.unsqueeze(1))
            return x_freq, index_tuple

        # |X_k| to find dominant components
        mag = x_freq.abs()
        vals, idx = torch.topk(mag, self.k, dim=1, largest=True, sorted=True)

        # Build tuple of indices for advanced indexing into x_freq
        b_idx, d_idx = torch.meshgrid(
            torch.arange(x_freq.size(0), device=x_freq.device),
            torch.arange(x_freq.size(2), device=x_freq.device),
            indexing='ij',
        )
        index_tuple = (b_idx.unsqueeze(1), idx, d_idx.unsqueeze(1))

        return x_freq[index_tuple], index_tuple

class LevelLayer(nn.Module):
    """
    Level‑update layer for ETS‑style decomposition.

    Parameters
    ----------
    dim : int
        Channel dimension of each input tensor (level / growth / season).
    output_dim : int
        Number of parallel level series (acts like “heads” in the smoothing layer).
    dropout : float, default 0.1
        Drop probability used inside the exponential‑smoothing module.
    """

    def __init__(self, dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim        = dim
        self.output_dim = output_dim

        # Exponential smoothing across `output_dim` independent series.
        self.es = ExponentialSmoothing(
            dim=1,                           # each series is scalar valued
            num_heads=output_dim,
            dropout=dropout,
            aux=True,                    # growth passes through aux channel
        )

        # Linear projections turning model‑space vectors into per‑series scalars.
        self.growth_proj = nn.Linear(dim, output_dim, bias=False)
        self.season_proj = nn.Linear(dim, output_dim, bias=False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        level:  torch.Tensor,   # [B, T, dim]
        growth: torch.Tensor,   # [B, T, dim]
        season: torch.Tensor,   # [B, T, dim]
    ) -> torch.Tensor:
        """
        Returns
        -------
        Tensor – updated level, shape **[B, T, output_dim]**
        """
        # Project growth & season to scalar‑per‑series and add singleton axis.
        growth = rearrange(self.growth_proj(growth), 'b t h -> b t h 1')
        season = rearrange(self.season_proj(season), 'b t h -> b t h 1')

        # Level is treated as one “series” per feature dimension.
        level  = rearrange(level, 'b t h -> b t h 1')

        # Exponential smoothing on deseasonalised level with growth as aux input.
        smoothed = self.es(level - season, aux_values=growth)  # [B, T, H, 1]

        # Return to shape [B, T, output_dim].
        return rearrange(smoothed, 'b t h d -> b t (h d)')


class EtsformerEncodeBlock(nn.Module):
    """
    Encoder block for ETSformer (level / trend / season decomposition).

    Parameters
    ----------
    dim : int
        Model channel width.
    num_heads : int
        Number of heads for the growth‑trend module.
    output_dim : int
        Number of future features to predict in the seasonal module.
    output_length : int
        Number of future time‑steps to extrapolate in the seasonal module.
    k : int | None
        Keep only the `k` dominant Fourier components (if not None).
    latent_dim : int, optional
        Hidden width inside the feed‑forward network.
        Defaults to `4 * dim` if omitted.
    dropout : float, default 0.1
        Drop probability used throughout the block.
    act_layer : Type[nn.Module], default nn.GELU
        Activation class used inside the feed‑forward layer.
    norm_layer : Type[nn.Module], default nn.LayerNorm
        Normalisation layer class.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        output_dim: int,
        output_length: int,
        k: Optional[int],
        latent_dim: Optional[int] = None,
        dropout: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        # ---- parameters ------------------------------------------------ #
        self.dim = dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.output_length = output_length
        self.k = k
        self.latent_dim = latent_dim or (4 * dim)

        # ---- sub‑modules -------------------------------------------------- #
        self.growth = GrowthLayer(dim, num_heads, dropout=dropout)
        self.season = FourierLayer(dim, output_length, k=k)
        self.level  = LevelLayer(dim, output_dim, dropout=dropout)
        self.ff     = FeedForward(
            dim,
            self.latent_dim,
            dropout=dropout,
            act_layer=act_layer,          # pass an *instance* (callable) to FeedForward
        )

        # ---- norm & dropout ---------------------------------------------- #
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    # -------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,                  # residual series  [B, T, dim]
        level: torch.Tensor,              # current level    [B, T, level_dim]
        attn_mask: Optional[torch.Tensor] = None,  # kept for API parity
    ):
        """
        Returns
        -------
        residual : Tensor  – [B, T, dim]
        level    : Tensor  – [B, T, level_dim]
        growth   : Tensor  – [B, T+1, H, D_h]
        season   : Tensor  – [B, T+output_dim, dim]
        """
        # 1. Seasonal component (adds `output_dim` future steps)
        season = self._season_block(x)                         # [B, T+output_dim, dim]

        x_ds   = x - season[:, :-self.output_length]              # deseasonalise

        # 2. Growth/trend component
        growth = self._growth_block(x_ds)                      # [B, T+1, H, D_h]
        x_dt   = self.norm1(x_ds - growth[:, 1:])              # detrend

        # 3. Position‑wise feed‑forward
        x_ff   = self.drop2(self.ff(x_dt))
        residual = self.norm2(x_dt + x_ff)

        # 4. Level update
        level = self.level(level, growth[:, :-1], season[:, :-self.output_length])

        return residual, level, growth, season

    # -------------------------------------------------------------------- #
    def _growth_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop1(self.growth(x))

    def _season_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.season(x)
        return self.drop2(x)


class DampingLayer(nn.Module):
    """
    Turns the **last single growth step** into a damped growth forecast
    of length `output_length`.

    Input shape
    -----------
    growth_step : Tensor **[B, 1, num_heads, head_dim]**

    Output shape
    ------------
    damped_growth : Tensor **[B, output_length, num_heads, head_dim]**

    Formula
    -------
    For each horizon step τ (1‑based):

        Ĝ_τ = G₀ * Σ_{i=1}^{τ} α^{i}

    where 0<α<1 is a learnable, per‑head damping factor.
    """

    def __init__(
        self,
        output_length: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_length = output_length
        self.num_heads     = num_heads

        # Learnable damping factor logits – one per head.
        self._damping_factor  = nn.Parameter(torch.randn(1, num_heads))

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  – shape **[B, 1, D]**

        Returns
        -------
        Tensor – shape **[B, L, D]**, where L == `output_length`
        """
        x = repeat(x, 'b 1 d -> b t d', t=self.output_length)
        b, t, d = x.shape

        powers = torch.arange(self.output_length).to(self._damping_factor.device) + 1
        powers = powers.view(self.output_length, 1)

        damping_factors = self._damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)

        x = x.view(b, t, self.num_heads, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)

        return x.view(b, t, d)

    # ------------------------------------------------------------------ #
    @property
    def alpha(self) -> torch.Tensor:
        """Damping factor α constrained to (0,1)."""
        return torch.sigmoid(self._logit_alpha)

class EtsformerDecodeBlock(nn.Module):
    """
    Decoder block for ETSformer.

    It converts the final **growth** tensor emitted by the encoder
    into a horizon‑length damped trend, and simply slices the most
    recent `output_length` steps from the seasonal tensor.

    Parameters
    ----------
    num_heads : int
        Number of heads in the growth tensor.
    output_length : int
        Forecast horizon (how many future time‑steps to generate).
    dropout : float, default 0.1
        Drop probability applied after the damping layer.
    """

    def __init__(
        self,
        num_heads: int,
        output_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads      = num_heads
        self.output_length  = output_length

        # Damps the very last growth slice and expands it to the horizon.
        self.growth_damping = DampingLayer(output_length, num_heads, dropout=dropout)
        self.dropout        = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        growth: torch.Tensor,   # [B, T+1, H, D_h] – full growth series
        season: torch.Tensor,   # [B, T+output_length, dim] – full seasonal series
    ):
        """
        Returns
        -------
        growth_horizon  : Tensor – damped growth forecast   [B, output_length, H, D_h]
        season_horizon  : Tensor – seasonal forecast        [B, output_length, dim]
        """
        # 1) Trend / growth horizon
        last_growth      = growth[:, -1:]                       # take final step
        growth_horizon   = self.dropout(self.growth_damping(last_growth))

        # 2) Seasonal horizon (last `output_length` steps)
        season_horizon   = season[:, -self.output_length:]

        return growth_horizon, season_horizon

if __name__ == '__main__':
    device = torch.device("cpu")

    # Example usage
    batch_size = 4
    seq_len = 64
    dim = 128
    num_heads = 4
    output_dim = 6
    output_length = 32
    k = 10
    latent_dim = 128
    dropout = 0.1

    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, dim).to(device)
    level = torch.randn(batch_size, seq_len, output_dim).to(device)

    # Create an instance of the EtsformerEncodeBlock
    model = EtsformerEncodeBlock(
        dim=dim,
        num_heads=num_heads,
        output_dim=output_dim,
        output_length=output_length,
        k=k,
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    # Forward pass
    residual, updated_level, growth, season = model(x, level)
    print("Residual shape:", residual.shape)  # Expected: [B, T, dim]
    print("Updated level shape:", updated_level.shape)  # Expected: [B, T, output_dim]
    print("Growth shape:", growth.shape)  # Expected: [B, T+1, H, D_h]
    print("Season shape:", season.shape)  # Expected: [B, T+output_dim, dim]

    # Create an instance of the EtsformerDecodeBlock
    model = EtsformerDecodeBlock(
        num_heads=num_heads,
        output_length=output_length,
        dropout=dropout
    ).to(device)
    # Forward pass for the decoder block
    growth_horizon, season_horizon = model(growth, season)
    print("Growth horizon shape:", growth_horizon.shape)  # Expected: [B, output_length, H, D_h]
    print("Season horizon shape:", season_horizon.shape)  # Expected: [B, output_length, dim]

