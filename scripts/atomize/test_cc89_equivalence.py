"""cc152 at initialisation must reproduce cc89 exactly on canonical protein.

cc152 widens cc89 in two places: 9 -> 22 input channels per atom, and 37 -> 61
atom slots. The surgery script copies cc89 into the (channel, slot) sub-block it
occupies and zeroes the rest, so feeding a canonical protein -- extra slots
empty, new channels whatever -- must give back cc89's predictions on the 37
canonical slots.

If this fails, the fine-tune is not starting from cc89 and any comparison
against it is meaningless.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from protpardelle.core import models
from protpardelle.data import atomize_io
from protpardelle.utils import dict_to_namespace, expand_env_in_config


def build(cfg_path: str, ckpt: str, sigma: float):
    d = expand_env_in_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    d["data"]["sigma_data"] = sigma
    d["data"]["auto_calc_sigma_data"] = False
    d["train"]["ckpt_path"] = ""
    m = models.Protpardelle(dict_to_namespace(d), device="cpu").eval()
    c = torch.load(ckpt, weights_only=False, map_location="cpu")
    sd = c["model_state_dict"] if isinstance(c, dict) else c.state_dict()
    m.load_state_dict(sd)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc89-config", default="examples/training/cc89.yaml")
    ap.add_argument(
        "--cc89-ckpt",
        default="/oak/stanford/groups/possu/tianyu/models/protpardelle-1c/weights/cc89_epoch415.pth",
    )
    ap.add_argument("--cc152-config", default="examples/training/cc152.yaml")
    ap.add_argument(
        "--cc152-ckpt",
        default="/scratch/users/tianyulu/protpardelle-1c-experiments/cc152_init/cc152_from_cc89_epoch415.pth",
    )
    ap.add_argument("--sigma-data", type=float, default=10.01)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    m89 = build(args.cc89_config, args.cc89_ckpt, args.sigma_data)
    m152 = build(args.cc152_config, args.cc152_ckpt, args.sigma_data)

    torch.manual_seed(0)
    b, n = 2, 64
    a37 = atomize_io.N_ATOM37
    a61 = atomize_io.N_SLOTS

    coords37 = torch.randn(b, n, a37, 3) * 10
    seq_mask = torch.ones(b, n)
    seq_mask[1, 50:] = 0
    ri = torch.arange(n).float().unsqueeze(0).repeat(b, 1)
    ci = torch.zeros(b, n)
    nl = torch.full((b, n), 1.0)
    cc37 = coords37 * 0.5
    ccm37 = torch.zeros(b, n, a37)
    ccm37[:, :10, :5] = 1

    def pad(x, dim):
        """Pad the atom-slot dim from 37 to 61 with zeros."""
        shape = list(x.shape)
        shape[dim] = a61 - a37
        return torch.cat([x, torch.zeros(shape, dtype=x.dtype)], dim=dim)

    with torch.no_grad():
        o89 = m89(
            noisy_coords=coords37, noise_level=nl, seq_mask=seq_mask,
            residue_index=ri, chain_index=ci,
            struct_crop_cond=cc37, crop_cond_mask=ccm37,
        )[0]
        o152 = m152(
            noisy_coords=pad(coords37, 2), noise_level=nl, seq_mask=seq_mask,
            residue_index=ri, chain_index=ci,
            struct_crop_cond=pad(cc37, 2), crop_cond_mask=pad(ccm37, 2),
            # Non-trivial atomized channels: their weights are zero, so these
            # must have no effect whatsoever.
            atom_elem=torch.randint(0, 9, (b, n, a61)),
            atom_ref=torch.randn(b, n, a61, 3) * 0.3,
            atom_noncanon=(torch.rand(b, n, a61) > 0.7).float(),
        )[0]

    canon_diff = (o89 - o152[:, :, :a37]).abs().max().item()
    extra_out = o152[:, :, a37:].abs().max().item()
    scale = o89.abs().mean().item()
    print(f"cc89 output scale              : {scale:.4f}")
    print(f"max |cc89 - cc152| canonical   : {canon_diff:.3e}  (rel {canon_diff / scale:.2e})")
    print(f"max |cc152| on empty extra slots: {extra_out:.3e}")
    ok = canon_diff / scale < args.tol and extra_out < 1e-3
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
