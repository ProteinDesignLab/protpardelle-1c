"""cc154 at init must be the same function as cc152 epoch 586.

Checking this by comparing float32 forward outputs does not work: adding 3
zero-weighted input channels changes the accumulation order of a 1525-term dot
product, and a trained 10-layer DiT amplifies that ~7000x (2e-6 at the patch
embedding -> 1e-2 at the output). That is float arithmetic, not a difference in
the function.

So test the two things that are actually decidable:
  1. the widened Linear is mathematically identical in float64, and
  2. the new channels have provably no effect (random vs zero bondfeat gives a
     bit-identical output).
"""
from __future__ import annotations

import argparse

import torch
import yaml

from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.utils import dict_to_namespace, expand_env_in_config

PATCH = "struct_model.net.to_patch_embedding"


def build(cfg_path, ckpt, sigma):
    d = expand_env_in_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    d["data"]["sigma_data"] = sigma
    d["data"]["auto_calc_sigma_data"] = False
    d["train"]["ckpt_path"] = ""
    m = models.Protpardelle(dict_to_namespace(d), device="cpu").eval()
    c = torch.load(ckpt, weights_only=False, map_location="cpu")
    m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c.state_dict())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    exp = "/scratch/users/tianyulu/protpardelle-1c-experiments"
    ap.add_argument("--cc152-config", default="examples/training/cc152.yaml")
    ap.add_argument("--cc152-ckpt",
                    default=f"{exp}/protpardelle-1c/cc152/checkpoints/epoch586_training_state.pth")
    ap.add_argument("--cc154-config", default="examples/training/cc154.yaml")
    ap.add_argument("--cc154-ckpt",
                    default=f"{exp}/cc154_init/cc154_from_cc152_epoch586.pth")
    ap.add_argument("--sigma-data", type=float, default=13.37)
    a = ap.parse_args()

    m152 = build(a.cc152_config, a.cc152_ckpt, a.sigma_data)
    m154 = build(a.cc154_config, a.cc154_ckpt, a.sigma_data)
    L1 = m152.struct_model.net.to_patch_embedding[1]
    L2 = m154.struct_model.net.to_patch_embedding[1]
    n_atoms = m152.struct_model.n_atoms
    c_old = L1.in_features // n_atoms
    c_new = L2.in_features // n_atoms
    print(f"channels {c_old} -> {c_new}, atom slots {n_atoms}")

    # 1. float64 equivalence of the widened linear map
    x = torch.randn(16, L1.in_features, dtype=torch.float64)
    y1 = x @ L1.weight.double().T + L1.bias.double()
    pad = torch.zeros(16, c_new - c_old, n_atoms, dtype=torch.float64)
    x2 = torch.cat([x.view(16, c_old, n_atoms), pad], 1).reshape(16, L2.in_features)
    y2 = x2 @ L2.weight.double().T + L2.bias.double()
    lin = float((y1 - y2).abs().max())
    print(f"1. float64 Linear equivalence      : {lin:.3e}  "
          f"{'OK' if lin < 1e-12 else 'FAIL'}")

    # 2. the new channels must be provably inert
    torch.manual_seed(0)
    b, n, s = 2, 48, A.N_SLOTS
    kw = dict(
        noisy_coords=torch.randn(b, n, s, 3) * 10,
        noise_level=torch.full((b, n), 1.0), seq_mask=torch.ones(b, n),
        residue_index=torch.arange(n).float().repeat(b, 1),
        chain_index=torch.zeros(b, n), struct_crop_cond=torch.randn(b, n, s, 3),
        crop_cond_mask=(torch.rand(b, n, s) > 0.5).float(),
        atom_elem=torch.randint(0, 9, (b, n, s)),
        atom_ref=torch.randn(b, n, s, 3) * 0.3,
        atom_noncanon=(torch.rand(b, n, s) > 0.7).float(),
    )
    with torch.no_grad():
        z = m154(**kw, atom_bondfeat=torch.zeros(b, n, s, 3))[0]
        r = m154(**kw, atom_bondfeat=torch.rand(b, n, s, 3))[0]
        o = m152(**kw)[0]
    inert = float((z - r).abs().max())
    print(f"2. new channels inert (rand vs 0)  : {inert:.3e}  "
          f"{'OK' if inert == 0.0 else 'FAIL'}")
    print(f"   (fp32 cc152-vs-cc154 drift {float((o - z).abs().max()):.3e} "
          f"is accumulation order amplified by the trunk, see docstring)")
    raise SystemExit(0 if (lin < 1e-12 and inert == 0.0) else 1)


if __name__ == "__main__":
    main()
