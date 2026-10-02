"""Initialise cc152 from cc89 weights.

cc152 is cc89 with 13 extra per-atom *input* channels ("atomized" conditioning).
Because the U-ViT flattens its input as ``(channel, atom)`` with channel-major
order, the cc89 patch-embedding weight is exactly the leading block of columns
of the cc152 one. Copy it there and zero the rest and the model is, at step 0,
numerically identical to cc89 on canonical protein input.

Every other parameter has an identical shape and is copied verbatim.

The output is a training_state dict (model_state_dict / optim_state_dict /
epoch / total_steps) so `train.ckpt_path` can pick it up with no code change.

Usage:
    python scripts/atomize/expand_cc89_to_cc152.py \
        --config examples/training/cc152.yaml \
        --cc89-ckpt /oak/.../weights/cc89_epoch415.pth \
        --out /scratch/.../cc152_from_cc89_epoch415.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from protpardelle.core import models
from protpardelle.utils import dict_to_namespace, expand_env_in_config

PATCH_W = "struct_model.net.to_patch_embedding.1.weight"
OUT_W = "struct_model.net.from_patch.1.weight"
OUT_B = "struct_model.net.from_patch.1.bias"
COND_W = "struct_model.net.cond_to_patch_embedding.1.weight"

# cc89: 9 input channels x 37 atom slots. cc152: 22 channels (9 + 9 element
# one-hot + 3 reference-conformer xyz + 1 extra-slot flag) x 61 atom slots.
N_CHAN_OLD, N_ATOMS_OLD = 9, 37
N_CHAN_NEW, N_ATOMS_NEW = 22, 61


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="examples/training/cc152.yaml")
    ap.add_argument("--chan-old", type=int, default=None,
                    help="override source channel count (cc152->cc154 is 22->25)")
    ap.add_argument("--atoms-old", type=int, default=None)
    ap.add_argument(
        "--cc89-ckpt",
        default="/oak/stanford/groups/possu/tianyu/models/protpardelle-1c/weights/cc89_epoch415.pth",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sigma-data",
        type=float,
        default=None,
        help="Override config sigma_data (shapes are unaffected either way).",
    )
    args = ap.parse_args()
    global N_CHAN_OLD, N_ATOMS_OLD, N_CHAN_NEW, N_ATOMS_NEW
    if args.chan_old is not None:
        N_CHAN_OLD = args.chan_old
    if args.atoms_old is not None:
        N_ATOMS_OLD = args.atoms_old

    with open(args.config, encoding="utf-8") as fh:
        config_dict = yaml.safe_load(fh)
    config_dict = expand_env_in_config(config_dict)
    if args.sigma_data is not None:
        config_dict["data"]["sigma_data"] = args.sigma_data
    config = dict_to_namespace(config_dict)

    model = models.Protpardelle(config, device="cpu")
    new_sd = model.state_dict()
    N_ATOMS_NEW = model.struct_model.n_atoms
    N_CHAN_NEW = new_sd[PATCH_W].shape[1] // N_ATOMS_NEW
    print(f"target model: {N_CHAN_NEW} channels x {N_ATOMS_NEW} atom slots")

    ckpt = torch.load(args.cc89_ckpt, weights_only=False, map_location="cpu")
    old_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt.state_dict()

    n_copied = n_expanded = n_skipped = 0
    missing = []
    for k, v_new in new_sd.items():
        if k not in old_sd:
            missing.append(k)
            continue
        v_old = old_sd[k]
        if v_old.shape == v_new.shape:
            new_sd[k] = v_old.clone()
            n_copied += 1
        elif k == PATCH_W and v_old.shape[0] == v_new.shape[0]:
            # Input patch embedding: [dim, n_channels * n_atoms]. The einops
            # flatten is "(c a)", i.e. index = c * n_atoms + a, so widening
            # n_atoms re-interleaves the columns -- a prefix copy would be
            # wrong. Map (c, a) -> (c, a) explicitly.
            w = torch.zeros_like(v_new)
            src = v_old.view(v_old.shape[0], N_CHAN_OLD, N_ATOMS_OLD)
            dst = w.view(w.shape[0], N_CHAN_NEW, N_ATOMS_NEW)
            dst[:, :N_CHAN_OLD, :N_ATOMS_OLD] = src
            new_sd[k] = dst.reshape(v_new.shape)
            n_expanded += 1
            print(
                f"expanded {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)} "
                f"[(c,a): ({N_CHAN_OLD},{N_ATOMS_OLD}) -> ({N_CHAN_NEW},{N_ATOMS_NEW})]"
            )
        elif k == OUT_W and v_old.shape[1] == v_new.shape[1]:
            # Output projection: [3 * n_atoms, dim], flattened as "(c a)" too.
            # New atom slots get zero rows, so at init the model predicts a
            # zero (pre-skip) update for every extra atom.
            w = torch.zeros_like(v_new)
            src = v_old.view(3, N_ATOMS_OLD, v_old.shape[1])
            dst = w.view(3, N_ATOMS_NEW, w.shape[1])
            dst[:, :N_ATOMS_OLD] = src
            new_sd[k] = dst.reshape(v_new.shape)
            n_expanded += 1
            print(f"expanded {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
        elif k == OUT_B and v_old.shape[0] != v_new.shape[0]:
            b = torch.zeros_like(v_new)
            b.view(3, N_ATOMS_NEW)[:, :N_ATOMS_OLD] = v_old.view(3, N_ATOMS_OLD)
            new_sd[k] = b
            n_expanded += 1
            print(f"expanded {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
        elif k == COND_W and v_old.shape[0] == v_new.shape[0]:
            # cond_to_patch_embedding: [time_cond_dim, 3 * n_atoms]
            w = torch.zeros_like(v_new)
            w.view(w.shape[0], 3, N_ATOMS_NEW)[:, :, :N_ATOMS_OLD] = v_old.view(
                v_old.shape[0], 3, N_ATOMS_OLD
            )
            new_sd[k] = w
            n_expanded += 1
            print(f"expanded {k}: {tuple(v_old.shape)} -> {tuple(v_new.shape)}")
        else:
            print(f"SHAPE MISMATCH, left at init: {k} {tuple(v_old.shape)} vs {tuple(v_new.shape)}")
            n_skipped += 1

    if missing:
        print(f"{len(missing)} params absent from cc89 (left at init): {missing[:8]}")
    extra = [k for k in old_sd if k not in new_sd]
    if extra:
        print(f"{len(extra)} cc89 params unused by cc152: {extra[:8]}")

    model.load_state_dict(new_sd)
    print(f"copied={n_copied} expanded={n_expanded} skipped={n_skipped}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "total_steps": 0,
        },
        out,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
