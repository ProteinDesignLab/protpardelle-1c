"""Neural network modules.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import copy
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float, Int
from torch import einsum
from torch.amp import autocast

from protpardelle.integrations.protein_mpnn import ProteinMPNN
from protpardelle.utils import get_logger, unsqueeze_trailing_dims

logger = get_logger(__name__)


def build_cyclic_harmonics(
    pair_dim: int, ring_size: int, device: torch.device
) -> Float[torch.Tensor, "D L"]:
    r"""Build cyclic harmonics for rotary embeddings.

    Constructs pair_dim harmonics k_i for a given ring size L, with frequencies
    \omega_i = 2\pi k_i / L.

    Uses a simple and robust "wrap-around" scheme to cover the [1, floor(L/2)] frequency band,
    avoiding aliasing.

    Args:
        pair_dim (int): Half of the feature dimension to be rotated.
        ring_size (int): Ring size.
        device (torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: Cyclic harmonics tensor of shape (pair_dim, ring_size).
    """

    max_k = max(1, ring_size // 2)

    k = torch.arange(1, pair_dim + 1, device=device)
    k = torch.ones_like(k) if max_k == 1 else 1 + (k - 1) % max_k

    return (2 * np.pi * k.float()) / ring_size


def compute_ring_size_per_token(
    cyclic_mask: Float[torch.Tensor, "B L"],
    chain_index: Int[torch.Tensor, "B L"] | None,
) -> Float[torch.Tensor, "B L"]:
    """Compute the ring size for each token in the sequence.

    For each token, return the cyclic length (ring_size) of its chain, computed
    as the number of True entries in cyclic_mask on that chain. Non-cyclic tokens receive 0.

    Args:
        cyclic_mask (torch.Tensor): Cyclic mask.
        chain_index (torch.Tensor | None, optional): Chain indices. Defaults to None.

    Returns:
        torch.Tensor: Ring size per token.
    """

    B = cyclic_mask.shape[0]
    device = cyclic_mask.device

    if chain_index is None:
        chain_index = torch.zeros_like(cyclic_mask, dtype=torch.long)
    else:
        chain_index = chain_index.long()  # ensure torch.long

    num_chains = torch.unique(chain_index).numel()
    assert (
        num_chains == torch.max(chain_index).item() + 1
    ), "chain_index should be 0-indexed, consecutive integers"

    # counts[b, c] = number of cyclic tokens in batch b that belong to chain c
    counts = torch.zeros(B, num_chains, dtype=torch.long, device=device)
    counts.scatter_add_(
        dim=1, index=chain_index, src=cyclic_mask.long()
    )  # batched bincount

    # ring_size[b, l] = counts[b, chain_index[b, l]]
    ring_size = counts.gather(dim=1, index=chain_index)

    # non-cyclic tokens should be 0
    ring_size = ring_size * cyclic_mask

    return ring_size


def circular_relpos_per_chain(
    residue_index: Int[torch.Tensor, "B L"],
    cyclic_mask: Float[torch.Tensor, "B L"] | None = None,
    chain_index: Int[torch.Tensor, "B L"] | None = None,
) -> Float[torch.Tensor, "B L L"]:
    """Signed circular relative positions per chain.

    - If (i, j) are on the same chain and both positions are cyclic:
        - Use the chain's cyclic length L to wrap the linear difference to the
        shortest signed arc in [-ceil(L/2)+1, +floor(L/2)] or [-floor(L/2), +ceil(L/2)-1].
        - In even rings, ties at L/2 are resolved by the sign of the original
        difference so that d_ij = -d_ji.
    - Otherwise (different chains or at least one non-cyclic): return the linear difference
        residue_index[i] - residue_index[j].

    Args:
        residue_index (torch.Tensor): Residue indices.
        cyclic_mask (torch.Tensor | None, optional): Cyclic mask. Defaults to None.
        chain_index (torch.Tensor | None, optional): Chain indices. Defaults to None.

    Returns:
        torch.Tensor: Circular relative positions.
    """

    # Use (i - j) to match the existing convention
    # different from the macrocycle-offset in RFpeptides/AfCycDesign
    dist = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)  # (B, L, L)

    if cyclic_mask is None:
        return dist

    if chain_index is None:
        chain_index = torch.zeros_like(residue_index, dtype=torch.long)  # (B, L)

    # Per-token ring size (0 for non-cyclic)
    ring_size_per_token = compute_ring_size_per_token(
        cyclic_mask=cyclic_mask, chain_index=chain_index
    )  # (B, L)

    # Cyclic pairs are same-chain AND both positions marked cyclic
    same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(-2)  # (B, L, L)
    cyclic_pair = (
        cyclic_mask.unsqueeze(-1) * cyclic_mask.unsqueeze(-2) * same_chain
    ).bool()

    # Avoid div/mod by zero: only wrap on cyclic pairs; 1 is a harmless placeholder elsewhere
    ring_size_pair = torch.where(
        cyclic_pair,
        ring_size_per_token.unsqueeze(-1),
        torch.ones_like(ring_size_per_token.unsqueeze(-1)),
    )  # (B, L, L)

    # Wrap to shortest signed arc
    # half = floor(L/2); keep values <= half as-is; values > half become negative
    half = ring_size_pair // 2  # floor(L/2)
    dist_mod = torch.remainder(dist, ring_size_pair)  # in [0, L-1]
    d_wrapped = torch.where(dist_mod <= half, dist_mod, dist_mod - ring_size_pair)

    # Range is now [-ceil(L/2)+1, +floor(L/2)] by construction.

    # Even rings: tie at L/2 -> use original sign to keep antisymmetry (d_ij = -d_ji)
    even = ring_size_pair % 2 == 0
    tie = even & (dist_mod == half)

    # dist < 0  -> +L/2 ;  dist >= 0 -> -L/2
    d_wrapped = torch.where(tie & (dist < 0), -half, d_wrapped)
    d_wrapped = torch.where(tie & (dist >= 0), half, d_wrapped)

    # Use wrapped values only on cyclic pairs; elsewhere keep linear differences
    d_ij = torch.where(cyclic_pair, d_wrapped, dist)  # (B, L, L)

    return d_ij


class RelativePositionalEncoding(nn.Module):
    def __init__(
        self,
        attn_dim: int = 8,
        max_rel_idx: int = 32,
        relchain: bool = False,
    ) -> None:
        super().__init__()
        self.max_rel_idx = max_rel_idx
        self.num_relpos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.num_relpos, attn_dim)
        self.relchain = relchain

    def forward(
        self,
        residue_index: Int[torch.Tensor, "B L"],
        cyclic_mask: Float[torch.Tensor, "B L"] | None = None,
        chain_index: Int[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L D"]:
        """Compute relative positional encodings.

        When cyclic_mask is given, automatically use circular relative positions
        for pairs within cyclic chains; otherwise, keep the original behavior.

        Args:
            residue_index (torch.Tensor): Residue indices.
            cyclic_mask (torch.Tensor | None, optional): Cyclic mask. Defaults to None.
            chain_index (torch.Tensor | None, optional): Chain indices. Defaults to None.

        Returns:
            torch.Tensor: Relative positional encodings.
        """

        if self.relchain:
            # Chain-level relative encoding: same/different chain binary encoding
            d_ij = (residue_index.unsqueeze(-1) != residue_index.unsqueeze(-2)).float()
        else:
            # Cyclic relative encoding
            d_ij = circular_relpos_per_chain(
                residue_index, cyclic_mask=cyclic_mask, chain_index=chain_index
            )

        device = d_ij.device

        v_bins = torch.arange(self.num_relpos, device=device) - self.max_rel_idx
        idxs = torch.abs(d_ij.unsqueeze(-1) - v_bins[None, None]).argmin(-1)
        p_ij = F.one_hot(idxs, num_classes=self.num_relpos).float()

        embeddings = self.linear(p_ij)  # (B, L, L, D)

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
) -> Float[torch.Tensor, "B H L D"]:
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

        if custom_freqs is not None:
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
        batch_size: int,
        residx: Float[torch.Tensor, "B L"],
        device: torch.device,
        offset: int = 0,
    ) -> Float[torch.Tensor, "B L"]:
        """Get sequence position for rotary embeddings depending on whether residue index is used or not.

        Args:
            seq_len (int): length of the sequence (for if not using residue index)
            batch_size (int): batch size
            residx (torch.Tensor): residue index (for if using residue index)
            device (torch.device): device to create the tensor on
            offset (int, optional): offset to add to the sequence position. Defaults to 0.

        Returns:
            seq_pos: sequence position tensor.
        """

        if self.use_residx:
            seq_pos = residx
        else:
            seq_pos = torch.arange(seq_len, device=device) + offset
            seq_pos = seq_pos.unsqueeze(0).expand(batch_size, -1)

        return seq_pos / self.interpolate_factor

    @autocast("cuda", enabled=False)
    def _forward(
        self,
        t: Float[torch.Tensor, "B L"],  # sequence positions
        chain_index: Float[torch.Tensor, "B L"] | None = None,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> Float[torch.Tensor, "B L D"]:
        should_cache = (
            self.cache_if_possible
            and (not self.learned_freq)
            and (seq_len is not None)
            and (self.freqs_for != "pixel")
        )

        if (
            should_cache
            and (self.cached_freqs is not None)
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

    @autocast("cuda", enabled=False)
    def forward(
        self,
        t: Float[torch.Tensor, "B L"],  # sequence positions
        chain_index: Int[torch.Tensor, "B L"] | None = None,
        cyclic_mask: Float[torch.Tensor, "B L"] | None = None,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> Float[torch.Tensor, "B L D"]:
        r"""Cyclic rotary embeddings.

        Return angle tensor matching the shape of self.forward (B, L, D). Tokens with
        cyclic_mask == 1.0 use ring harmonics \omega_i = 2\pi k_i / L_{\text{chain}};
        all other tokens keep the standard RoPE angles.
        """

        # Compute standard angles first to preserve the baseline behaviour for non-cyclic tokens
        std_freqs = self._forward(
            t, chain_index=chain_index, seq_len=seq_len, offset=offset
        )  # (B, L, D)

        if cyclic_mask is None:
            return std_freqs

        is_cyclic = cyclic_mask > 0  # boolean mask of cyclic positions

        if not is_cyclic.any():
            return std_freqs

        D = std_freqs.shape[-1]
        assert D % 2 == 0, "feature dimension must be multiple of 2 for RoPE"
        pair_dim = D // 2
        device = std_freqs.device

        # Compute the ring size per token (zero for non-cyclic positions)
        ring_size_per_token = compute_ring_size_per_token(
            cyclic_mask=cyclic_mask, chain_index=chain_index
        )  # (B, L)

        # Start from the standard angles and overwrite cyclic tokens in-place
        mix_freqs = std_freqs.clone()
        # Reduce positions modulo ring size for cyclic tokens (stability & correctness)
        # Note: clamp(min=1) avoids mod-by-zero on non-cyclic tokens (masked out anyway).
        divisor = ring_size_per_token.clamp(min=1)
        if t.is_floating_point():
            divisor = divisor.to(t.dtype)
        t_mod = torch.where(is_cyclic, torch.remainder(t, divisor), t)

        unique_ring_size_list = torch.unique(ring_size_per_token[is_cyclic]).tolist()
        for ring_size in unique_ring_size_list:
            sel = (
                ring_size_per_token == ring_size
            ) & is_cyclic  # Tokens in this batch whose ring size equals ring_size
            if not sel.any():
                continue
            omega = build_cyclic_harmonics(
                pair_dim, ring_size=ring_size, device=device
            )  # \omega_i
            # \theta = position * \omega, where position = t[b, sel]
            theta_pairs = t_mod[sel].unsqueeze(-1) * omega
            theta_full = repeat(theta_pairs, "n p -> n (p r)", r=2)
            mix_freqs[sel] = theta_full

        return mix_freqs  # (B, L, D)


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

    def forward(self, x: Float[torch.Tensor, "B L"]) -> Float[torch.Tensor, "B L "]:
        """Forward pass for noise embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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


def get_value_or_default(value: Any, default: Any) -> Any:
    """Get the value or default.

    Args:
        value (Any): The value to check.
        default (Any): The default value to return if value is None.

    Returns:
        Any: The original value if not None, otherwise the default value.
    """

    return value if value is not None else default


def posemb_sincos_1d(
    patches: Float[torch.Tensor, "B L D"],
    temperature: float = 10000.0,
    residue_index: Int[torch.Tensor, "B L"] | None = None,
) -> Float[torch.Tensor, "B L D"]:
    """1D sine-cosine positional embedding."""

    _, L, D = patches.shape
    device = patches.device

    if residue_index is None:
        residue_index = torch.arange(L, device=device)

    if D % 2 != 0:
        raise ValueError("feature dimension must be multiple of 2 for sincos emb")

    omega = torch.arange(D // 2, device=device) / (D // 2 - 1)
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
        num_cyclic_heads: int = 0,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        dit: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = get_value_or_default(dim_context, dim)

        self.time_cond = None

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.dit = dit

        if time_cond_dim is not None:
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
        if motif_cond_dim is not None:
            self.motif_cond = nn.Sequential(
                nn.SiLU(), nn.Linear(motif_cond_dim, dim * 2)
            )

            nn.init.zeros_(self.motif_cond[-1].weight)
            nn.init.zeros_(self.motif_cond[-1].bias)

            # Add gating
            self.motif_gate = nn.Linear(motif_cond_dim, dim * 2)
            self.sigmoid = nn.Sigmoid()

            nn.init.zeros_(self.motif_gate.weight)
            nn.init.zeros_(self.motif_gate.bias)

        self.scale = dim_head ** (-0.5)
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        if attn_bias_dim is not None:
            self.attn_bias_proj = nn.Sequential(
                Rearrange("b d i j -> b i j d"),
                nn.Linear(attn_bias_dim, heads),
                Rearrange("b i j h -> b h i j"),
            )
        else:
            self.attn_bias_proj = None

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        if rotary_embedding_module is None:
            self.use_rope = False
        else:
            self.use_rope = True
            self.rope = rotary_embedding_module

        # Reserve the heads that will use the cyclic rotation (first num_cyclic_heads heads)
        self.num_cyclic_heads = num_cyclic_heads
        head_mask = torch.zeros(heads)  # (H,)
        if self.num_cyclic_heads > 0:
            head_mask[: self.num_cyclic_heads] = 1.0
        self.register_buffer("cyclic_head_mask", head_mask, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "B L D"],
        residx: Float[torch.Tensor, "B L"],
        context: Float[torch.Tensor, "B L D"] | None = None,
        time: Float[torch.Tensor, "B L"] | None = None,
        motif: Float[torch.Tensor, "B M D"] | None = None,
        attn_bias: Float[torch.Tensor, "B H L L"] | None = None,
        seq_mask: Float[torch.Tensor, "B L"] | None = None,
        chain_index: Float[torch.Tensor, "B L"] | None = None,
        cyclic_mask: Float[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L D"]:

        context = get_value_or_default(context, x)

        x = self.norm(x)

        if time is not None:
            if self.dit:
                scale, shift, alpha_1 = self.time_cond(time).chunk(3, dim=-1)
            else:
                scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if motif is not None:
            motif_scale_shift = self.sigmoid(self.motif_gate(motif)) * self.motif_cond(
                motif
            )
            scale, shift = motif_scale_shift.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if context is not None:
            context = self.norm_context(context)

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = q * self.scale

        B, _, L, _ = q.shape
        device = q.device

        if self.use_rope:
            # Standard RoPE path
            seq_pos = self.rope.get_seq_pos(
                seq_len=L,
                batch_size=B,
                residx=residx,
                device=device,
                offset=0,
            )
            std_freqs = self.rope.forward(
                seq_pos, seq_len=L, chain_index=chain_index
            )  # (B, L, D)

            q_std = apply_rotary_emb(std_freqs, q)
            k_std = apply_rotary_emb(std_freqs, k)

            # When cyclic_mask is provided, compute cyclic RoPE angles and enable them only on the cyclic heads
            if self.num_cyclic_heads and (cyclic_mask is not None):
                cyc_freqs = self.rope.forward(
                    seq_pos,
                    chain_index=chain_index,
                    cyclic_mask=cyclic_mask,
                    seq_len=L,
                    offset=0,
                )
                q_cyc = apply_rotary_emb(cyc_freqs, q)
                k_cyc = apply_rotary_emb(cyc_freqs, k)

                # Head-level routing: first num_cyclic_heads heads use cyclic rotation, others stay standard
                cyclic_head_mask = self.cyclic_head_mask.view(1, -1, 1, 1)
                q = q_std + cyclic_head_mask * (q_cyc - q_std)
                k = k_std + cyclic_head_mask * (k_cyc - k_std)
            else:
                q, k = q_std, k_std

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # If cyclic heads are active and cyclic_mask exists, mask out pairs outside the same ring for those heads
        if self.use_rope and self.num_cyclic_heads and (cyclic_mask is not None):
            assert chain_index is not None  # TODO: support chain_index=None case
            same_chain = chain_index.unsqueeze(-1) == chain_index.unsqueeze(
                -2
            )  # (B, L, L)
            cyc_pair = (
                cyclic_mask.unsqueeze(-1) * cyclic_mask.unsqueeze(-2) * same_chain
            ).bool()  # (B, L, L)

            row_is_cyc = (
                cyclic_mask.unsqueeze(-1).expand_as(cyc_pair).bool()
            )  # (B, L, L)
            bad_rows = (~cyc_pair) & row_is_cyc  # (B, L, L)

            head_mask = self.cyclic_head_mask.view(1, -1, 1, 1)  # (1, H, 1, 1)
            sim = sim.masked_fill(head_mask.bool() & bad_rows.unsqueeze(1), -torch.inf)

        if attn_bias is not None:
            if self.attn_bias_proj is not None:
                attn_bias = self.attn_bias_proj(attn_bias)
            assert attn_bias is not None
            sim = sim + attn_bias

        if seq_mask is not None:
            attn_mask = torch.einsum("b i, b j -> b i j", seq_mask, seq_mask).unsqueeze(
                1
            )
            sim = sim.masked_fill(~attn_mask.bool(), -torch.inf)
        attn = sim.softmax(dim=-1)

        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if self.dit and (time is not None):
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

        if time_cond_dim is not None:
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

        if motif_cond_dim is not None:
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

        if time is not None:
            if self.dit:
                scale, shift, alpha_2 = self.time_cond(time).chunk(3, dim=-1)
            else:
                scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if motif is not None:
            motif_scale_shift = self.sigmoid(self.motif_gate(motif)) * self.motif_cond(
                motif
            )
            scale, shift = motif_scale_shift.chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if self.dit:
            x = self.linear_in(x)
            x = self.nonlinearity(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear_out(x)

        if self.dit and (time is not None):
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
        num_cyclic_heads: int = 0,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit: bool = False,
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
                    attn_dim=heads, max_rel_idx=position_embedding_max, relchain=False
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
            num_cyclic_heads=num_cyclic_heads,
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
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        motif: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
        chain_index: torch.Tensor | None = None,
        cyclic_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Drop-in replacement.
        Interface notes:
        1) If `pos_emb_type` contains "relchain", add the same-versus-different-chain bias.
        2) If `pos_emb_type` contains "relative", add the residue_index-based relative bias.
            - With both cyclic_mask and chain_index, use circular distances for pairs on the same ring
                and keep linear offsets elsewhere (see RelativePositionalEncoding.forward).
        3) Accumulate the resulting positional bias into attn_bias.
        4) Pass cyclic_mask to each attention layer via a temporary attribute for cyclic RoPE and masking.
        """

        if self.pos_emb_type == "absolute":
            pos_emb = posemb_sincos_1d(x)
            x = x + pos_emb
        elif self.pos_emb_type == "absolute_residx":
            assert residue_index is not None
            pos_emb = posemb_sincos_1d(x, residue_index=residue_index)
            x = x + pos_emb
        else:
            # Accumulate positional bias without overriding an existing attn_bias
            pos_bias = None

            # 1) Optional chain-level bias (same vs different chain)
            if "relchain" in self.pos_emb_type:
                if chain_index is None:
                    raise ValueError("relchain positional bias requires `chain_index`.")
                chain_bias = self.relchain(
                    chain_index
                )  # Expected shape: (B, H, L, L) or broadcastable to that shape
                pos_bias = chain_bias if pos_bias is None else (pos_bias + chain_bias)

            # 2) Relative positional bias; wrap distances per ring when cyclic_mask is provided
            if "relative" in self.pos_emb_type:
                if residue_index is None:
                    raise ValueError(
                        "relative positional bias requires `residue_index`."
                    )
                assert isinstance(self.relpos, nn.Sequential)

                # Submodule 0 is RelativePositionalEncoding, which consumes index/chain_index/cyclic_mask
                # Submodule 1 rearranges to (B, heads, L, L)
                rel_core = self.relpos[0]
                rel_rearr = self.relpos[1]
                rel_bias_core = rel_core(
                    residue_index, chain_index=chain_index, cyclic_mask=cyclic_mask
                )  # (B, L, L, heads)
                relpos_bias = rel_rearr(rel_bias_core)  # (B, heads, L, L)

                pos_bias = relpos_bias if pos_bias is None else (pos_bias + relpos_bias)

            # Merge the accumulated pos_bias into the overall attn_bias
            if pos_bias is not None:
                attn_bias = pos_bias if attn_bias is None else (attn_bias + pos_bias)

        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1)

        # Core loop: expose cyclic_mask to the attention layer
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
                cyclic_mask=cyclic_mask,
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
        num_cyclic_heads: int = 0,
        noise_residual: bool = False,  # Not used
        ssadj_cond: bool = False,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit: bool = False,
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
            num_cyclic_heads=num_cyclic_heads,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout,
            ff_dropout=ff_dropout,
            dit=dit,
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
        time_cond: Float[torch.Tensor, "B L A T"] | None = None,
        motif_cond: Float[torch.Tensor, "B L A M"] | None = None,
        pair_bias: Float[torch.Tensor, "B L A A"] | None = None,
        seq_mask: Float[torch.Tensor, "B L A 1"] | None = None,
        residue_index: Float[torch.Tensor, "B L"] | None = None,
        chain_index: Float[torch.Tensor, "B L"] | None = None,
        cyclic_mask: Float[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L A D"]:

        if self.num_conv_layers > 0:  # pad up to even dims
            coords = F.pad(coords, (0, 0, 0, 0, 0, 1, 0, 0))

        x = rearr_coords = rearrange(coords, "b l a x -> b x l a")
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
            cyclic_mask=cyclic_mask,
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
