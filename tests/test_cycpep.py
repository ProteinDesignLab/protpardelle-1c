import math

import pytest
import torch

# Adjust to your module path
from protpardelle.core.modules import (
    build_cyclic_harmonics,
    circular_relpos_per_chain,
    compute_ring_size_per_token,
)


# ------------------------------
# Helpers to build simple cases
# ------------------------------
def mk_single_chain(L: int, cyclic: bool = True):
    """B=1, one chain, residue_index = 0..L-1, chain_index = 0."""
    ri = torch.arange(L)[None, :]  # (1, L)
    ci = torch.zeros(1, L, dtype=torch.long)  # (1, L)
    cm = (
        torch.ones(1, L, dtype=torch.bool)
        if cyclic
        else torch.zeros(1, L, dtype=torch.bool)
    )
    return ri, ci, cm


def mk_two_chains(L1: int, L2: int, cyclic_first=True, cyclic_second=False):
    """B=1, two concatenated chains with ids 0 and 1; residue indices are within-chain."""
    ri = torch.tensor(list(range(L1)) + list(range(L2)))[None, :]  # (1, L1+L2)
    ci = torch.tensor([0] * L1 + [1] * L2, dtype=torch.long)[None, :]
    cm = torch.zeros_like(ci, dtype=torch.bool)
    if cyclic_first:
        cm[:, :L1] = True
    if cyclic_second:
        cm[:, L1:] = True
    return ri, ci, cm


# ------------------------------
# build_cyclic_harmonics
# ------------------------------
@pytest.mark.parametrize("L", [1, 2, 3, 4, 7, 10, 11])
@pytest.mark.parametrize("pair_dim", [1, 2, 4, 8])
def test_build_cyclic_harmonics_band_and_wrap(L, pair_dim):
    w = build_cyclic_harmonics(
        pair_dim=pair_dim, ring_size=L, device=torch.device("cpu")
    )
    assert w.shape == (pair_dim,), "Expected (pair_dim,) vector of angular freqs"
    max_k = max(1, L // 2)
    allowed = (2 * math.pi / L) * torch.arange(1, max_k + 1)
    for val in w:
        assert torch.isclose((val - allowed).abs().min(), torch.tensor(0.0), atol=1e-6)
    if pair_dim > max_k and max_k > 1:
        for i in range(pair_dim - max_k):
            assert torch.isclose(w[i], w[i % max_k], atol=1e-6)


# ------------------------------
# compute_ring_size_per_token
# ------------------------------
def test_compute_ring_size_per_token_vectorized_single_batch():
    L1, L2 = 6, 4
    _, ci, cm = mk_two_chains(L1, L2, cyclic_first=True, cyclic_second=False)
    ring = compute_ring_size_per_token(ci, cm)
    assert ring.shape == (1, L1 + L2)
    assert torch.all(ring[0, :L1] == L1)  # fully cyclic chain
    assert torch.all(ring[0, L1:] == 0)  # non-cyclic chain


def test_compute_ring_size_per_token_vectorized_batched_same_L():
    # Use same total length in both batch items to satisfy torch.cat shape rules.
    _, ci0, cm0 = mk_two_chains(
        6, 4, cyclic_first=True, cyclic_second=False
    )  # total L=10
    _, ci1, cm1 = mk_single_chain(10, cyclic=True)  # total L=10
    ci = torch.cat([ci0, ci1], dim=0)
    cm = torch.cat([cm0, cm1], dim=0)
    ring = compute_ring_size_per_token(ci, cm)
    # batch 0 checks
    assert torch.all(ring[0, :6] == 6)
    assert torch.all(ring[0, 6:10] == 0)
    # batch 1 checks
    assert torch.all(ring[1, :] == 10)


# ------------------------------
# circular_relpos_per_chain
# ------------------------------
def test_noncyclic_is_linear():
    L = 8
    ri, ci, cm = mk_single_chain(L, cyclic=False)
    D = circular_relpos_per_chain(ri, ci, cm)[0]
    expect = ri[0].unsqueeze(-1) - ri[0].unsqueeze(-2)  # (j - i)
    assert torch.equal(D, expect)


def test_odd_L_shortest_arc_and_antisym():
    L = 9
    ri, ci, cm = mk_single_chain(L, cyclic=True)
    D = circular_relpos_per_chain(ri, ci, cm)[0]
    lo, hi = -(L // 2), (L // 2)  # [-4, +4]
    assert int(D.min()) >= lo and int(D.max()) <= hi
    assert torch.all(D + D.T == 0)


def test_even_L_tie_rule_antisymmetry():
    L = 10
    ri, ci, cm = mk_single_chain(L, cyclic=True)
    D = circular_relpos_per_chain(ri, ci, cm)[0]
    assert D[0].tolist() == [0, -1, -2, -3, -4, -5, 4, 3, 2, 1]
    assert D[0, 5].item() == -L // 2 and D[5, 0].item() == +L // 2
    assert torch.all(D + D.T == 0)


def test_multichain_cyclic_vs_linear_and_offblocks():
    L1, L2 = 6, 4
    ri, ci, cm = mk_two_chains(L1, L2, cyclic_first=True, cyclic_second=False)
    D = circular_relpos_per_chain(ri, ci, cm)[0]

    # Top-left cyclic block is antisymmetric
    assert torch.all(D[:L1, :L1] + D[:L1, :L1].T == 0)

    # Bottom-right block is linear (j - i)
    expect_lin = ri[0, L1:].unsqueeze(1) - ri[0, L1:].unsqueeze(
        0
    )  # (1,L2) - (L2,1) -> (L2,L2)
    assert torch.equal(D[L1:, L1:], expect_lin)

    # Off-diagonal blocks are linear with correct orientation:
    # Rows: i in chain 0; Cols: j in chain 1 -> (1,L2) - (L1,1) -> (L1,L2)
    expect_01 = ri[0, :L1].unsqueeze(1) - ri[0, L1:].unsqueeze(0)
    assert torch.equal(D[:L1, L1:], expect_01)

    # Rows: i in chain 1; Cols: j in chain 0 -> (1,L1) - (L2,1) -> (L2,L1)
    expect_10 = ri[0, L1:].unsqueeze(1) - ri[0, :L1].unsqueeze(0)
    assert torch.equal(D[L1:, :L1], expect_10)


def test_batched_equivalence_cyclic_blocks_same_L():
    # Both batch items of total length 10 to allow torch.cat on dim=0
    ri0, ci0, cm0 = mk_single_chain(10, cyclic=True)
    ri1, ci1, cm1 = mk_two_chains(6, 4, cyclic_first=True, cyclic_second=True)
    ri = torch.cat([ri0, ri1], dim=0)
    ci = torch.cat([ci0, ci1], dim=0)
    cm = torch.cat([cm0, cm1], dim=0)

    D_b = circular_relpos_per_chain(ri, ci, cm)
    D0 = circular_relpos_per_chain(ri0, ci0, cm0)
    D1 = circular_relpos_per_chain(ri1, ci1, cm1)
    assert torch.equal(D_b[0], D0[0])
    assert torch.equal(D_b[1], D1[0])


# ------------------------------
# Device smoke test
# ------------------------------
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_device_smoke(device):
    L = 10
    ri, ci, cm = mk_single_chain(L, cyclic=True)
    ri, ci, cm = ri.to(device), ci.to(device), cm.to(device)
    D = circular_relpos_per_chain(ri, ci, cm)
    assert D.shape == (1, L, L)
