"""Neural network modules.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float, Int
from torch import broadcast_tensors, einsum
from torch.amp import autocast
from torch.optim.lr_scheduler import LRScheduler

from protpardelle.common import residue_constants
from protpardelle.integrations.protein_mpnn import ProteinMPNN
from protpardelle.utils import get_logger, unsqueeze_trailing_dims

logger = get_logger(__name__)


def circular_relpos(
    index: torch.Tensor,
    cyc_mask: torch.Tensor | None = None,
    ring_size: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute shortest signed circular relative positions on a ring.

    Force d_ij to be in [-floor(L/2), ..., +ceil(L/2)-1]. If L is even,
    ties at +L/2 are mapped to -L/2 for a unique convention.

    Args:
        index (torch.Tensor): Absolute positions. (B, N)
        cyc_mask (torch.Tensor | None, optional): Sequence mask for cyclic peptides.
            Defaults to None. (B, N)
        ring_size (int | torch.Tensor | None, optional): Ring size. Defaults to None. (B,)

    Returns:
        torch.Tensor: Circular relative positions. (B, N, N)
    """

    B, N = index.shape
    device = index.device

    if ring_size is None:
        L = (
            cyc_mask.sum(dim=-1)
            if cyc_mask is not None
            else torch.full((B,), N, device=device)
        )
    elif isinstance(ring_size, int):
        L = torch.full((B,), ring_size, device=device)
    elif isinstance(ring_size, torch.Tensor):
        assert ring_size.shape == (B,)
        L = ring_size.to(device)
    else:
        raise ValueError("ring_size must be None, int, or torch.Tensor of shape (B,)")
    L = L.view(B, 1, 1)  # (B, 1, 1) for broadcasting

    # Pairwise raw differences of absolute positions
    d_ij = index.unsqueeze(-1) - index.unsqueeze(-2)  # (B, N, N)

    # Wrap to shortest signed distance on the ring of size L (per batch)
    half = L // 2
    d_ij = ((d_ij + half) % L) - half

    # For even L: map +L/2 -> -L/2 to make the range symmetric and unique
    even = L % 2 == 0
    d_ij = torch.where(even & (d_ij == half), d_ij - L, d_ij)

    return d_ij


class RelativePositionalEncoding(nn.Module):
    def __init__(
        self,
        attn_dim: int = 8,
        max_rel_idx: int = 32,
        relchain: bool = False,
        cyclic: bool = False,
    ) -> None:
        super().__init__()

        self.max_rel_idx = max_rel_idx
        self.num_relpos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.num_relpos, attn_dim)
        self.relchain = relchain
        self.cyclic = cyclic

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encodings.

        Args:
            index (torch.Tensor): Absolute positions. (B, L)

        Returns:
            torch.Tensor: Relative positional encodings. (B, L, L, C)
        """

        if self.relchain:
            if self.cyclic:
                logger.warning(
                    "Cyclic relative positions are only supported for residues; will be ignored here."
                )
            d_ij = (index.unsqueeze(-1) != index.unsqueeze(-2)).float()
        elif self.cyclic:
            d_ij = circular_relpos(index)
        else:
            d_ij = index.unsqueeze(-1) - index.unsqueeze(-2)
        device = d_ij.device

        v_bins = torch.arange(self.num_relpos, device=device) - self.max_rel_idx
        idxs = torch.abs(d_ij.unsqueeze(-1) - v_bins[None, None]).argmin(-1)
        p_ij = F.one_hot(idxs, num_classes=self.num_relpos).float()
        embeddings = self.linear(p_ij)

        return embeddings


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Args:
        x (torch.Tensor): Input tensor where r=2. (..., D*R)

    Returns:
        torch.Tensor: Rotated tensor where r=2. (..., D*R)
    """

    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)

    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: Float[torch.Tensor, "B L D"],
    t: Float[torch.Tensor, "B H L D"],
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[:, -seq_len:].to(t)

    freqs = freqs.unsqueeze(-3)  # add attn head dimension
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    return torch.cat((t_left, t, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_residx: bool,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
        num_freqs: int = 1,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: int = 512,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = True,
    ) -> None:
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        # Use residue index
        self.use_residx = (
            use_residx  # use residue index rather than position in sequence
        )
        if use_residx:
            assert (
                not cache_if_possible
            ), "Caching is not supported with residue index since each sequence will have different positions."

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * np.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs)
        else:
            raise ValueError(
                f"freqs_for must be one of 'lang', 'pixel', 'constant', or you can pass in custom freqs directly. Got {freqs_for}"
            )

        self.cache_if_possible = cache_if_possible
        self.dim = dim
        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device
        self.tmp_store("dummy", torch.tensor(0))

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = xpos_scale_base
            self.tmp_store("scale", scale)
        else:
            self.tmp_store("scale", None)
        self.use_xpos = use_xpos

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def tmp_store(self, key: str, value: torch.Tensor | None) -> None:
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(
        self,
        seq_len: int,
        residx: Float[torch.Tensor, "B L"],
        B: int,
        device,
        dtype,
        offset=0,
    ) -> Float[torch.Tensor, "B L"]:
        """
        Get sequence position for rotary embeddings depending on whether residue index is used or not.
        - seq_len: length of the sequence (for if not using residue index)
        - residx: residue index (for if using residue index)
        - B: batch size
        """
        if self.use_residx:
            seq_pos = residx
        else:
            seq_pos = torch.arange(seq_len, device=device, dtype=dtype) + offset
            seq_pos = seq_pos.unsqueeze(0).expand(B, -1)
        return seq_pos / self.interpolate_factor

    def rotate_queries_or_keys(
        self,
        t: Float[torch.Tensor, "B H L D"],
        residx: Int[torch.Tensor, "B L"],
        chain_index: Int[torch.Tensor, "B L"] | None,
        seq_dim=None,
        offset=0,
    ):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert (
            not self.use_xpos
        ), "you must use .rotate_queries_and_keys method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len, B = t.device, t.dtype, t.shape[seq_dim], t.shape[0]

        seq_pos = self.get_seq_pos(
            seq_len=seq_len,
            residx=residx,
            B=B,
            device=device,
            dtype=dtype,
            offset=offset,
        )

        freqs = self.forward(
            seq_pos, seq_len=seq_len, offset=offset, chain_index=chain_index
        )

        if seq_dim == -3:
            freqs = rearrange(freqs, "b n d -> b n 1 d")

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def get_scale(self, t: torch.Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if (
            should_cache
            and exists(self.cached_scales)
            and (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(
        self,
        t: Float[torch.Tensor, "B L"],  # sequence positions
        chain_index: Float[torch.Tensor, "B L"] | None = None,
        seq_len=None,
        offset=0,
    ) -> Float[torch.Tensor, "B L N"]:
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
        )

        if (
            should_cache
            and exists(self.cached_freqs)
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        if chain_index is not None:
            # Combined frequency
            residue_freqs = self.freqs
            chain_freqs = residue_freqs[::2]
            residue_freqs = einsum(
                "..., f -> ... f", t.type(residue_freqs.dtype), residue_freqs
            )
            chain_freqs = einsum(
                "..., f -> ... f", chain_index.type(chain_freqs.dtype), chain_freqs
            )

            residue_indices = torch.arange(self.dim // 2) % residue_freqs.shape[-1]
            chain_indices = torch.arange(self.dim // 2) // residue_freqs.shape[-1]
            combined_freqs = (
                residue_freqs[:, :, residue_indices] + chain_freqs[:, :, chain_indices]
            )
            freqs = repeat(combined_freqs, "... f -> ... (f r)", r=2)
        else:
            freqs = self.freqs
            freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


########################################
# Adapted from https://github.com/NVlabs/edm


class NoiseEmbedding(nn.Module):
    """Noise embedding layer."""

    def __init__(
        self, num_channels: int, max_positions: int = 10000, endpoint: bool = False
    ) -> None:
        """Noise embedding layer.

        Args:
            num_channels (int): Number of channels in the input.
            max_positions (int, optional): Maximum number of positions. Defaults to 10000.
            endpoint (bool, optional): Whether to include endpoint. Defaults to False.
        """

        super().__init__()

        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for noise embedding.

        Args:
            x (torch.Tensor): Input tensor. (B, N)

        Returns:
            torch.Tensor: Output tensor. (B, N, C)
        """

        device = x.device
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float, device=device
        )
        freqs = freqs / (self.num_channels // 2 - self.endpoint)
        freqs = (1 / self.max_positions) ** freqs

        if x.ndim == 1:
            x = x.outer(freqs.to(x.dtype))
        else:
            b, l = x.shape
            d = freqs.shape[0]
            x = x.flatten().outer(freqs.to(x.dtype)).view(b, l, d)
        x = torch.cat([x.cos(), x.sin()], dim=-1)

        return x


########################################
# Adapted from github.com/lucidrains
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://github.com/lucidrains/recurrent-interface-network-pytorch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def posemb_sincos_1d(
    patches: torch.Tensor,
    temperature: float = 10000.0,
    residue_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D sine-cosine positional embedding."""

    B, L, C = patches.shape
    device = patches.device

    if residue_index is None:
        residue_index = torch.arange(L, device=device)

    if C % 2 != 0:
        raise ValueError("feature dimension must be multiple of 2 for sincos emb")

    omega = torch.arange(C // 2, device=device) / (C // 2 - 1)
    omega = 1.0 / (temperature**omega)

    residue_index = residue_index.unsqueeze(-1) * omega
    posemb = torch.cat((residue_index.sin(), residue_index.cos()), dim=-1)

    return posemb


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class NoiseConditioningBlock(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            NoiseEmbedding(num_in_channels),
            nn.Linear(num_in_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, num_out_channels),
        )

    def forward(self, noise_level):
        ret = self.block(noise_level)
        if len(noise_level.shape) == 1:
            ret = rearrange(ret, "b d -> b 1 d")
        return ret


class TimeCondResnetBlock(nn.Module):
    def __init__(
        self, nic, noc, cond_nc, conv_layer=nn.Conv2d, dropout=0.1, num_norm_in_groups=4
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=nic // num_norm_in_groups, num_channels=nic),
            nn.SiLU(),
            conv_layer(nic, noc, 3, 1, 1),
        )
        self.cond_proj = nn.Linear(cond_nc, noc * 2)
        self.mid_norm = nn.GroupNorm(num_groups=noc // 4, num_channels=noc)
        self.dropout = dropout if dropout is None else nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=noc // 4, num_channels=noc),
            nn.SiLU(),
            conv_layer(noc, noc, 3, 1, 1),
        )
        self.mismatch = False
        if nic != noc:
            self.mismatch = True
            self.conv_match = conv_layer(nic, noc, 1, 1, 0)

    def forward(self, x, time=None):
        h = self.block1(x)

        if time is not None:
            h = self.mid_norm(h)
            scale, shift = self.cond_proj(time).chunk(2, dim=-1)
            h = (h * (unsqueeze_trailing_dims(scale, h) + 1)) + unsqueeze_trailing_dims(
                shift, h
            )

        if self.dropout is not None:
            h = self.dropout(h)

        h = self.block2(h)

        if self.mismatch:
            x = self.conv_match(x)

        return x + h


class TimeCondAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_context: int | None = None,
        heads: int = 4,
        dim_head: int = 32,
        norm: bool = False,
        norm_context: bool = False,
        time_cond_dim: int | None = None,
        motif_cond_dim: int | None = None,
        attn_bias_dim: int | None = None,
        rotary_embedding_module: RotaryEmbedding | None = None,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        dit: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.dit = dit

        if exists(time_cond_dim):
            if self.dit:
                self.time_cond = nn.Sequential(
                    nn.SiLU(), nn.Linear(time_cond_dim, dim * 3)
                )
            else:
                self.time_cond = nn.Sequential(
                    nn.SiLU(), nn.Linear(time_cond_dim, dim * 2)
                )

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        # add motif conditioning track
        if exists(motif_cond_dim):
            self.motif_cond = nn.Sequential(
                nn.SiLU(), nn.Linear(motif_cond_dim, dim * 2)
            )

            nn.init.zeros_(self.motif_cond[-1].weight)
            nn.init.zeros_(self.motif_cond[-1].bias)

            # add gating
            self.motif_gate = nn.Linear(motif_cond_dim, dim * 2)
            self.sigmoid = nn.Sigmoid()

            nn.init.zeros_(self.motif_gate.weight)
            nn.init.zeros_(self.motif_gate.bias)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_bias_proj = None
        if attn_bias_dim is not None:
            self.attn_bias_proj = nn.Sequential(
                Rearrange("b a i j -> b i j a"),
                nn.Linear(attn_bias_dim, heads),
                Rearrange("b i j a -> b a i j"),
            )

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        if rotary_embedding_module is None:
            self.use_rope = False
        else:
            self.use_rope = True
            self.rope = rotary_embedding_module

    def forward(
        self,
        x,
        residx: Float[torch.Tensor, "B L"],
        context=None,
        time=None,
        motif=None,
        attn_bias=None,
        seq_mask=None,
        chain_index=None,
    ):
        # attn_bias is b, c, i, j
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        x = self.norm(x)

        if exists(time):
            if self.dit:
                scale, shift, alpha_1 = self.time_cond(time).chunk(3, dim=-1)
            else:
                scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if exists(motif):
            motif_scale_shift = self.sigmoid(self.motif_gate(motif)) * self.motif_cond(
                motif
            )
            scale, shift = motif_scale_shift.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if has_context:
            context = self.norm_context(context)

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        q = q * self.scale

        if self.use_rope:
            q = self.rope.rotate_queries_or_keys(q, residx, chain_index=chain_index)
            k = self.rope.rotate_queries_or_keys(k, residx, chain_index=chain_index)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_bias is not None:
            if self.attn_bias_proj is not None:
                attn_bias = self.attn_bias_proj(attn_bias)
            sim += attn_bias
        if seq_mask is not None:
            attn_mask = torch.einsum("b i, b j -> b i j", seq_mask, seq_mask).unsqueeze(
                1
            )
            sim -= (1 - attn_mask) * 1e6
        attn = sim.softmax(dim=-1)

        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if self.dit and exists(time):
            out = out * (alpha_1 + 1)

        if seq_mask is not None:
            out = out * seq_mask.unsqueeze(-1)

        out = self.out_dropout(out)

        return out


class TimeCondFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dim_out=None,
        time_cond_dim=None,
        motif_cond_dim=None,
        dropout=0.1,
        dit=False,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm = LayerNorm(dim)

        self.time_cond = None
        self.dropout = None
        self.dit = dit
        inner_dim = int(dim * mult)

        if exists(time_cond_dim):
            if self.dit:
                self.time_cond = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_cond_dim, dim * 3),
                )
            else:
                self.time_cond = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_cond_dim, inner_dim * 2),
                )

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        if exists(motif_cond_dim):
            # add motif conditioning track
            self.motif_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(motif_cond_dim, inner_dim * 2),
            )

            nn.init.zeros_(self.motif_cond[-1].weight)
            nn.init.zeros_(self.motif_cond[-1].bias)

            # add gating
            self.motif_gate = nn.Linear(motif_cond_dim, inner_dim * 2)
            self.sigmoid = nn.Sigmoid()

            nn.init.zeros_(self.motif_gate.weight)
            nn.init.zeros_(self.motif_gate.bias)

        self.linear_in = nn.Linear(dim, inner_dim)
        self.nonlinearity = nn.SiLU()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(inner_dim, dim_out)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, time=None, motif=None):
        x = self.norm(x)

        if not self.dit:
            x = self.linear_in(x)
            x = self.nonlinearity(x)

        if exists(time):
            if self.dit:
                scale, shift, alpha_2 = self.time_cond(time).chunk(3, dim=-1)
            else:
                scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if exists(motif):
            motif_scale_shift = self.sigmoid(self.motif_gate(motif)) * self.motif_cond(
                motif
            )
            scale, shift = motif_scale_shift.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if self.dit:
            x = self.linear_in(x)
            x = self.nonlinearity(x)

        if exists(self.dropout):
            x = self.dropout(x)

        x = self.linear_out(x)

        if self.dit and exists(time):
            x = x * (alpha_2 + 1)

        return x


class TimeCondTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        time_cond_dim: int,
        motif_cond_dim: int | None = None,
        attn_bias_dim: int | None = None,
        mlp_inner_dim_mult: int = 4,
        position_embedding_type: Literal[
            "rotary",
            "rotary_relchain",
            "absolute",
            "absolute_residx",
            "relative",
            "relative_relchain",
            "none",
        ] = "rotary",
        position_embedding_max: int = 32,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit: bool = False,
        cyclic: bool = False,
    ) -> None:
        super().__init__()

        self.rope = None
        self.pos_emb_type = position_embedding_type

        if self.pos_emb_type not in {
            "rotary",
            "rotary_relchain",
            "absolute",
            "absolute_residx",
            "relative",
            "relative_relchain",
            "none",
        }:
            raise ValueError(f"Unknown position embedding type {self.pos_emb_type}")

        if "rotary" in position_embedding_type:
            self.rope = RotaryEmbedding(
                dim=(dim // heads), use_residx=True, cache_if_possible=False
            )  # changed to use residx
        if "relative" in position_embedding_type:
            self.relpos = nn.Sequential(
                RelativePositionalEncoding(
                    attn_dim=heads, max_rel_idx=position_embedding_max, cyclic=cyclic
                ),
                Rearrange("b i j d -> b d i j"),
            )
        if "relchain" in position_embedding_type:
            self.relchain = nn.Sequential(
                RelativePositionalEncoding(
                    attn_dim=heads, max_rel_idx=position_embedding_max, relchain=True
                ),
                Rearrange("b i j d -> b d i j"),
            )

        time_cond_attention = TimeCondAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            norm=True,
            time_cond_dim=time_cond_dim,
            motif_cond_dim=motif_cond_dim,
            attn_bias_dim=attn_bias_dim,
            rotary_embedding_module=self.rope,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout,
            dit=dit,
        )
        time_cond_feed_forward = TimeCondFeedForward(
            dim,
            mlp_inner_dim_mult,
            time_cond_dim=time_cond_dim,
            motif_cond_dim=motif_cond_dim,
            dropout=ff_dropout,
            dit=dit,
        )
        layer = nn.ModuleList(
            [
                time_cond_attention,
                time_cond_feed_forward,
            ]
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(depth)])

    def forward(
        self,
        x: Float[torch.Tensor, "B L D"],
        time=None,
        motif=None,
        attn_bias=None,
        context=None,
        seq_mask=None,
        residue_index=None,
        chain_index=None,
    ):
        if self.pos_emb_type == "absolute":
            pos_emb = posemb_sincos_1d(x)
            x = x + pos_emb
        elif self.pos_emb_type == "absolute_residx":
            assert residue_index is not None
            pos_emb = posemb_sincos_1d(x, residue_index=residue_index)
            x = x + pos_emb
        if "relchain" in self.pos_emb_type:
            assert chain_index is not None
            pos_emb = self.relchain(chain_index)
            attn_bias = pos_emb if attn_bias is None else attn_bias + pos_emb
        if "relative" in self.pos_emb_type:
            assert residue_index is not None
            if "pos_emb" in locals():  # TODO: avoid this
                pos_emb += self.relpos(residue_index)
            else:
                pos_emb = self.relpos(residue_index)
            attn_bias = pos_emb if attn_bias is None else attn_bias + pos_emb
        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1)

        # Begin transformer layers
        for attn, ff in self.layers:
            x = x + attn(
                x,
                residx=residue_index,
                context=context,
                time=time,
                motif=motif,
                attn_bias=attn_bias,
                seq_mask=seq_mask,
                chain_index=chain_index,
            )
            x = x + ff(x, time=time, motif=motif)

            if seq_mask is not None:
                x = x * seq_mask.unsqueeze(-1)

        return x


class TimeCondUViT(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        num_filt_per_layer: list[int] = [],  # TODO: tuple
        num_blocks_per_layer: int = 2,
        num_atoms: int = 37,
        channels_per_atom: int = 6,
        attn_bias_dim: int | None = None,
        time_cond_dim: int | None = None,
        motif_cond_dim: int | None = None,
        conv_skip_connection: bool = False,
        position_embedding_type: str = "rotary",
        position_embedding_max: int = 32,
        noise_residual: bool = False,  # Not used
        ssadj_cond: bool = False,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit: bool = False,
        cyclic: bool = False,
    ) -> None:
        super().__init__()

        # Initialize configuration params
        if time_cond_dim is None:
            time_cond_dim = dim * 4

        self.position_embedding_type = position_embedding_type
        channels = channels_per_atom
        self.num_conv_layers = num_conv_layers = len(num_filt_per_layer)
        if num_conv_layers > 0:
            post_conv_filt = num_filt_per_layer[-1]
        self.conv_skip_connection = conv_skip_connection and num_conv_layers == 1
        transformer_seq_len = seq_len // (2**num_conv_layers)
        assert transformer_seq_len % patch_size == 0

        dim_a = post_conv_atom_dim = max(1, num_atoms // (2 ** (num_conv_layers - 1)))
        if num_conv_layers == 0:
            patch_dim = patch_size * num_atoms * channels_per_atom
            patch_dim_out = patch_size * num_atoms * 3
            dim_a = num_atoms
        elif conv_skip_connection and num_conv_layers == 1:
            patch_dim = patch_size * (channels + post_conv_filt) * post_conv_atom_dim
            patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim
        elif num_conv_layers > 0:
            patch_dim = patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim

        # Make downsampling conv
        # Downsamples n-1 times where n is num_conv_layers
        down_conv = []
        block_in = channels
        for i, nf in enumerate(num_filt_per_layer):
            block_out = nf
            layer = []
            for j in range(num_blocks_per_layer):
                num_groups = 2 if i == 0 and j == 0 else 4
                layer.append(
                    TimeCondResnetBlock(
                        block_in,
                        block_out,
                        time_cond_dim,
                        num_norm_in_groups=num_groups,
                    )
                )
                block_in = block_out
            down_conv.append(nn.ModuleList(layer))
        self.down_conv = nn.ModuleList(down_conv)

        # Make transformer
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=patch_size),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )
        self.cond_to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=patch_size),
            nn.Linear(patch_size * num_atoms * 3, time_cond_dim),
            LayerNorm(time_cond_dim),
        )

        if ssadj_cond:
            self.adj_to_embedding = nn.Sequential(
                nn.Embedding(2, dim),
                nn.Linear(dim, heads),
                LayerNorm(heads),
                Rearrange("b i j d -> b d i j"),
            )

        self.transformer = TimeCondTransformer(
            dim,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            motif_cond_dim=motif_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
            position_embedding_max=position_embedding_max,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout,
            ff_dropout=ff_dropout,
            dit=dit,
            cyclic=cyclic,
        )
        self.from_patch = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, patch_dim_out),
            Rearrange("b n (p c a) -> b c (n p) a", p=patch_size, a=dim_a),
        )
        nn.init.zeros_(self.from_patch[-2].weight)
        nn.init.zeros_(self.from_patch[-2].bias)

        # Make upsampling conv
        up_conv = []
        for i, nf in enumerate(reversed(num_filt_per_layer)):
            skip_in = nf
            block_out = nf
            layer = []
            for _ in range(num_blocks_per_layer):
                layer.append(
                    TimeCondResnetBlock(block_in + skip_in, block_out, time_cond_dim)
                )
                block_in = block_out
            up_conv.append(nn.ModuleList(layer))
        self.up_conv = nn.ModuleList(up_conv)

        # Conv out
        if num_conv_layers > 0:
            self.conv_out = nn.Sequential(
                nn.GroupNorm(num_groups=block_out // 4, num_channels=block_out),
                nn.SiLU(),
                nn.Conv2d(block_out, channels // 2, 3, 1, 1),
            )

    def forward(
        self,
        coords: Float[torch.Tensor, "B L A 3"],
        time_cond,
        motif_cond=None,
        pair_bias=None,
        seq_mask=None,
        residue_index=None,
        chain_index=None,
    ) -> torch.Tensor:

        if self.num_conv_layers > 0:  # pad up to even dims
            coords = F.pad(coords, (0, 0, 0, 0, 0, 1, 0, 0))

        x = rearr_coords = rearrange(coords, "b n a c -> b c n a")
        hidden_states = []
        for i, layer in enumerate(self.down_conv):
            for block in layer:
                x = block(x, time=time_cond)
                hidden_states.append(x)
            if i != self.num_conv_layers - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        if self.conv_skip_connection:
            x = torch.cat([x, rearr_coords], 1)

        x = self.to_patch_embedding(x)

        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x = x * unsqueeze_trailing_dims(seq_mask, x)

        attn_bias = None
        if pair_bias is not None:
            attn_bias = self.adj_to_embedding(pair_bias)

        x = self.transformer(
            x,
            time=time_cond,
            motif=motif_cond,
            attn_bias=attn_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        x = self.from_patch(x)

        for i, layer in enumerate(self.up_conv):
            for block in layer:
                x = torch.cat([x, hidden_states.pop()], 1)
                x = block(x, time=time_cond)
            if i != self.num_conv_layers - 1:
                x = F.interpolate(x, size=hidden_states[-1].shape[2:], mode="nearest")

        if self.num_conv_layers > 0:
            x = self.conv_out(x)
            x = x[..., :-1, :]  # drop even-dims padding

        x = rearrange(x, "b c n a -> b n a c")

        return x


########################################


class LinearWarmupCosineDecay(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        min_lr: float = 1e-6,
        **kwargs,
    ) -> None:

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + decay_steps

        super().__init__(optimizer, **kwargs)  # call init at the end

    def get_lr(self) -> list[float]:
        """Compute the current learning rate for all param groups."""

        # Handle warmup (if any)
        if (self.warmup_steps > 0) and (self.last_epoch < self.warmup_steps):
            progress = self.last_epoch / max(1, self.warmup_steps)
            curr_lr = progress * self.max_lr
        # Cosine decay phase
        elif (self.decay_steps > 0) and (self.last_epoch < self.total_steps):
            # Fraction of decay completed (0 at start of decay, 1 at end)
            decay_progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.decay_steps
            )
            time = decay_progress * np.pi
            curr_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1.0 + float(np.cos(time))
            )
        else:
            curr_lr = self.min_lr

        # Return LR for each param group
        return [curr_lr for _ in self.optimizer.param_groups]


class NoiseConditionalProteinMPNN(nn.Module):
    def __init__(
        self,
        num_channels: int = 128,
        num_layers: int = 3,
        num_neighbors: int = 32,
        vocab_size: int = 21,
        time_cond_dim: int | None = None,
        input_S_is_embeddings: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.vocab_size = vocab_size
        self.time_cond_dim = time_cond_dim
        self.bb_idxs_if_atom37 = [
            residue_constants.atom_order[atom] for atom in ["N", "CA", "C", "O"]
        ]
        self.ca_idxs_if_atom37 = [residue_constants.atom_order[atom] for atom in ["CA"]]

        self.mpnn = ProteinMPNN(
            num_letters=vocab_size,
            node_features=num_channels,
            edge_features=num_channels,
            hidden_dim=num_channels,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            vocab=vocab_size,
            k_neighbors=num_neighbors,
            augment_eps=0.0,
            dropout=0.1,
            ca_only=True,  # CHANGED -- better to use CA-only for noisy coords
            time_cond_dim=time_cond_dim,
            input_S_is_embeddings=input_S_is_embeddings,
        )

    def forward(
        self,
        denoised_coords: torch.Tensor,
        noisy_aatype: torch.Tensor,
        seq_mask: torch.Tensor,
        residue_index: torch.Tensor,
        time_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if denoised_coords.shape[-2] == 37:
            denoised_coords = denoised_coords[:, :, 1]  # CHANGED to ca-only

        node_embs, encoder_embs = self.mpnn(
            X=denoised_coords,
            S=noisy_aatype,
            mask=seq_mask,
            chain_M=seq_mask,
            residue_idx=residue_index,
            chain_encoding_all=seq_mask,
            randn=None,
            use_input_decoding_order=False,
            decoding_order=None,
            causal_mask=False,
            time_cond=time_cond,
            return_node_embs=True,
        )

        return node_embs, encoder_embs
