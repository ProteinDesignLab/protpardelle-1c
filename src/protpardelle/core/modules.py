"""Neural network modules.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import broadcast_tensors, einsum
from torch.amp import autocast
from torchtyping import TensorType

from protpardelle import utils
from protpardelle.common import residue_constants
from protpardelle.integrations import protein_mpnn

########################################
# Adapted from https://github.com/ermongroup/ddim


def downsample(x):
    return F.avg_pool2d(x, 2, 2, ceil_mode=True)


def upsample_coords(x, shape):
    new_l, new_w = shape
    return F.interpolate(x, size=(new_l, new_w), mode="nearest")


########################################
# Adapted from https://github.com/aqlaboratory/openfold


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.contiguous().permute(first_inds + [zero_index + i for i in inds])


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions.unsqueeze(-2) - all_atom_positions.unsqueeze(-3)) ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos.unsqueeze(-2) - all_atom_pred_pos.unsqueeze(-3)) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


class RelativePositionalEncoding(nn.Module):
    def __init__(self, attn_dim=8, max_rel_idx=32, relchain=False):
        super().__init__()
        self.max_rel_idx = max_rel_idx
        self.n_rel_pos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.n_rel_pos, attn_dim)
        self.relchain = relchain

    def forward(self, index):
        if self.relchain:
            d_ij = torch.clamp(
                torch.abs(index.unsqueeze(-1) - index.unsqueeze(-2)), min=0, max=1
            )
        else:
            d_ij = index.unsqueeze(-1) - index.unsqueeze(-2)
        v_bins = torch.arange(self.n_rel_pos).to(d_ij.device) - self.max_rel_idx
        idxs = (d_ij.unsqueeze(-1) - v_bins[None, None]).abs().argmin(-1)
        p_ij = F.one_hot(idxs, num_classes=self.n_rel_pos)
        embeddings = self.linear(p_ij.float())
        return embeddings


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(
    freqs: TensorType["b n d"],
    t: TensorType["b h n d"],
    start_index=0,
    scale=1.0,
    seq_dim=-2,
):
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
        dim,
        use_residx: bool,
        custom_freqs: torch.Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
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
                cache_if_possible is False
            ), "Caching is not supported with residue index since each sequence will have different positions."

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

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

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(
        self,
        seq_len: int,
        residx: TensorType["b n", float],
        B: int,
        device,
        dtype,
        offset=0,
    ) -> TensorType["b n", float]:
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
        t: TensorType["b h n d"],
        residx: TensorType["b n", int],
        chain_index: TensorType["b n", int] | None,
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

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(
            q, seq_dim=seq_dim, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(
        self, q, k, residx: TensorType["b n", float], seq_dim=None
    ):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, residx, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

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
        t: TensorType["b n", float],  # sequence positions
        chain_index: TensorType["b n", float] | None = None,
        seq_len=None,
        offset=0,
    ) -> TensorType["b n f", float]:
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


class Noise_Embedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        if len(x.shape) == 1:
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


def posemb_sincos_1d(patches, temperature=10000, residue_index=None):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device) if residue_index is None else residue_index
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.unsqueeze(-1) * omega
    pe = torch.cat((n.sin(), n.cos()), dim=-1)
    return pe.type(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class NoiseConditioningBlock(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super().__init__()
        self.block = nn.Sequential(
            Noise_Embedding(n_in_channel),
            nn.Linear(n_in_channel, n_out_channel),
            nn.SiLU(),
            nn.Linear(n_out_channel, n_out_channel),
        )

    def forward(self, noise_level):
        ret = self.block(noise_level)
        if len(noise_level.shape) == 1:
            ret = rearrange(ret, "b d -> b 1 d")
        return ret


class TimeCondResnetBlock(nn.Module):
    def __init__(
        self, nic, noc, cond_nc, conv_layer=nn.Conv2d, dropout=0.1, n_norm_in_groups=4
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=nic // n_norm_in_groups, num_channels=nic),
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
            h = (
                h * (utils.unsqueeze_trailing_dims(scale, h) + 1)
            ) + utils.unsqueeze_trailing_dims(shift, h)

        if self.dropout is not None:
            h = self.dropout(h)

        h = self.block2(h)

        if self.mismatch:
            x = self.conv_match(x)

        return x + h


class TimeCondAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        heads=4,
        dim_head=32,
        norm=False,
        norm_context=False,
        time_cond_dim=None,
        motif_cond_dim=None,
        attn_bias_dim=None,
        rotary_embedding_module=None,
        attn_dropout=0.0,
        out_dropout=0.0,
        dit=False,
    ):
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

        self.use_rope = False
        if rotary_embedding_module is not None:
            self.use_rope = True
            self.rope = rotary_embedding_module

    def forward(
        self,
        x,
        residx: TensorType["b n", float],
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

        if x.shape[-1] != self.norm.gamma.shape[-1]:
            print(context.shape, x.shape, self.norm.gamma.shape)

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
        dim,
        depth,
        heads,
        dim_head,
        time_cond_dim,
        motif_cond_dim=None,
        attn_bias_dim=None,
        mlp_inner_dim_mult=4,
        position_embedding_type: str = "rotary",
        position_embedding_max: int = 32,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit=False,
    ):
        super().__init__()

        self.rope = None
        self.pos_emb_type = position_embedding_type

        assert self.pos_emb_type in [
            "rotary",
            "rotary_relchain",
            "absolute",
            "absolute_residx",
            "relative",
            "relative_relchain",
            "none",
        ], f"Unknown position embedding type {self.pos_emb_type}"

        if "rotary" in position_embedding_type:
            self.rope = RotaryEmbedding(
                dim=(dim // heads), use_residx=True, cache_if_possible=False
            )  # Changed to use residx
        if "relative" in position_embedding_type:
            self.relpos = nn.Sequential(
                RelativePositionalEncoding(
                    attn_dim=heads, max_rel_idx=position_embedding_max
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

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TimeCondAttention(
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
                        ),
                        TimeCondFeedForward(
                            dim,
                            mlp_inner_dim_mult,
                            time_cond_dim=time_cond_dim,
                            motif_cond_dim=motif_cond_dim,
                            dropout=ff_dropout,
                            dit=dit,
                        ),
                    ]
                )
            )

    def forward(
        self,
        x: TensorType["b n d", float],
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
            if "pos_emb" in locals():
                pos_emb += self.relpos(residue_index)
            else:
                pos_emb = self.relpos(residue_index)
            attn_bias = pos_emb if attn_bias is None else attn_bias + pos_emb
        if seq_mask is not None:
            x = x * seq_mask.unsqueeze(-1)

        # Begin transformer layers
        for i, (attn, ff) in enumerate(self.layers):
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
        *,
        seq_len: int,
        dim: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_filt_per_layer: list[int] = [],
        n_blocks_per_layer: int = 2,
        n_atoms: int = 37,
        channels_per_atom: int = 6,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        motif_cond_dim: int = None,
        conv_skip_connection: bool = False,
        position_embedding_type: str = "rotary",
        position_embedding_max: int = 32,
        noise_residual: bool = False,
        ssadj_cond: bool = False,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        dit: bool = False,
    ):
        super().__init__()

        # Initialize configuration params
        if time_cond_dim is None:
            time_cond_dim = dim * 4
        # if motif_cond_dim is None:
        #     motif_cond_dim = dim * 4
        self.position_embedding_type = position_embedding_type
        channels = channels_per_atom
        self.n_conv_layers = n_conv_layers = len(n_filt_per_layer)
        if n_conv_layers > 0:
            post_conv_filt = n_filt_per_layer[-1]
        self.conv_skip_connection = conv_skip_connection and n_conv_layers == 1
        transformer_seq_len = seq_len // (2**n_conv_layers)
        assert transformer_seq_len % patch_size == 0
        num_patches = transformer_seq_len // patch_size
        dim_a = post_conv_atom_dim = max(1, n_atoms // (2 ** (n_conv_layers - 1)))
        if n_conv_layers == 0:
            patch_dim = patch_size * n_atoms * channels_per_atom
            patch_dim_out = patch_size * n_atoms * 3
            dim_a = n_atoms
        elif conv_skip_connection and n_conv_layers == 1:
            patch_dim = patch_size * (channels + post_conv_filt) * post_conv_atom_dim
            patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim
        elif n_conv_layers > 0:
            patch_dim = patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim

        # Make downsampling conv
        # Downsamples n-1 times where n is n_conv_layers
        down_conv = []
        block_in = channels
        for i, nf in enumerate(n_filt_per_layer):
            block_out = nf
            layer = []
            for j in range(n_blocks_per_layer):
                n_groups = 2 if i == 0 and j == 0 else 4
                layer.append(
                    TimeCondResnetBlock(
                        block_in, block_out, time_cond_dim, n_norm_in_groups=n_groups
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
            nn.Linear(patch_size * n_atoms * 3, time_cond_dim),
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
        for i, nf in enumerate(reversed(n_filt_per_layer)):
            skip_in = nf
            block_out = nf
            layer = []
            for j in range(n_blocks_per_layer):
                layer.append(
                    TimeCondResnetBlock(block_in + skip_in, block_out, time_cond_dim)
                )
                block_in = block_out
            up_conv.append(nn.ModuleList(layer))
        self.up_conv = nn.ModuleList(up_conv)

        # Conv out
        if n_conv_layers > 0:
            self.conv_out = nn.Sequential(
                nn.GroupNorm(num_groups=block_out // 4, num_channels=block_out),
                nn.SiLU(),
                nn.Conv2d(block_out, channels // 2, 3, 1, 1),
            )

    def forward(
        self,
        coords: TensorType["b n a x", float],
        time_cond,
        motif_cond=None,
        pair_bias=None,
        seq_mask=None,
        residue_index=None,
        chain_index=None,
        return_emb=False,
    ):

        if self.n_conv_layers > 0:  # pad up to even dims
            coords = F.pad(coords, (0, 0, 0, 0, 0, 1, 0, 0))

        x = rearr_coords = rearrange(coords, "b n a c -> b c n a")
        hiddens = []
        for i, layer in enumerate(self.down_conv):
            for block in layer:
                x = block(x, time=time_cond)
                hiddens.append(x)
            if i != self.n_conv_layers - 1:
                x = downsample(x)

        if self.conv_skip_connection:
            x = torch.cat([x, rearr_coords], 1)

        x = self.to_patch_embedding(x)

        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x = x * utils.unsqueeze_trailing_dims(seq_mask, x)

        attn_bias = None
        if pair_bias is not None:
            attn_bias = self.adj_to_embedding(pair_bias)

        emb = self.transformer(
            x,
            time=time_cond,
            motif=motif_cond,
            attn_bias=attn_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
        )

        x = self.from_patch(emb)

        for i, layer in enumerate(self.up_conv):
            for block in layer:
                x = torch.cat([x, hiddens.pop()], 1)
                x = block(x, time=time_cond)
            if i != self.n_conv_layers - 1:
                x = upsample_coords(x, hiddens[-1].shape[2:])

        if self.n_conv_layers > 0:
            x = self.conv_out(x)
            x = x[..., :-1, :]  # drop even-dims padding

        x = rearrange(x, "b c n a -> b n a c")

        if return_emb:
            return x, emb
        else:
            return x


########################################


class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_steps=1000,
        decay_steps=int(1e6),
        min_lr=1e-6,
        **kwargs,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + decay_steps
        super(LinearWarmupCosineDecay, self).__init__(optimizer, **kwargs)

    def get_lr(self):
        # TODO double check for off-by-one errors
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.last_epoch / self.warmup_steps * self.max_lr
            return [curr_lr for group in self.optimizer.param_groups]
        elif self.last_epoch < self.total_steps:
            time = (self.last_epoch - self.warmup_steps) / self.decay_steps * np.pi
            curr_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + np.cos(time)
            )
            return [curr_lr for group in self.optimizer.param_groups]
        else:
            return [self.min_lr for group in self.optimizer.param_groups]


class NoiseConditionalProteinMPNN(nn.Module):
    def __init__(
        self,
        n_channel=128,
        n_layers=3,
        n_neighbors=32,
        time_cond_dim=None,
        vocab_size=21,
        input_S_is_embeddings=False,
    ):
        super().__init__()
        self.n_channel = n_channel
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.time_cond_dim = time_cond_dim
        self.vocab_size = vocab_size
        self.bb_idxs_if_atom37 = [
            residue_constants.atom_order[a] for a in ["N", "CA", "C", "O"]
        ]
        self.ca_idxs_if_atom37 = [residue_constants.atom_order[a] for a in ["CA"]]

        self.mpnn = protein_mpnn.ProteinMPNN(
            num_letters=vocab_size,
            node_features=n_channel,
            edge_features=n_channel,
            hidden_dim=n_channel,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            vocab=vocab_size,
            k_neighbors=n_neighbors,
            augment_eps=0.0,
            dropout=0.1,
            ca_only=True,  # CHANGED -- better to use CA-only for noisy coords
            time_cond_dim=time_cond_dim,
            input_S_is_embeddings=input_S_is_embeddings,
        )

    def forward(
        self, denoised_coords, noisy_aatype, seq_mask, residue_index, time_cond
    ):
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
