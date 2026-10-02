"""Measure how well a model denoises non-canonical atoms.

The question cc152 has to answer is: when a structure contains a modified amino
acid, a cofactor or a metal, can the model place those atoms? cc89 cannot even
see them -- they are dropped by the atom37 loader -- so the honest comparison is
"same atomized input, cc152 weights vs cc89 weights". Because cc152's new input
channels are zero-initialised from cc89, running the *initial* cc152 checkpoint
is exactly running cc89.

For a grid of noise levels we noise the clean coordinates, run a single
denoising pass, and report RMSD between the denoised and clean coordinates,
broken out by atom class:

    bb          backbone N/CA/C/O of canonical residues
    sc          canonical side-chain atoms
    ncaa_extra  the extra atoms of a modified amino acid (the acetyl of ALY,
                the carbamate of KCX, the retinal of LYR, ...)
    hetero      ligand / cofactor / metal pseudo-residue atoms

Usage:
    python scripts/atomize/eval_atomized.py \
        --ckpt  .../cc152/checkpoints/epoch12_training_state.pth \
        --base  .../cc152_init/cc152_from_cc89_epoch415.pth \
        --keys  .../val_atomized_pdb_keys.list --n 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from protpardelle.common import residue_constants
from protpardelle.core import diffusion, models
from protpardelle.data import atomize_io
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config

BB_SLOTS = [residue_constants.atom_order[a] for a in ("N", "CA", "C", "O")]
UNK = residue_constants.restype_num


def load_model(config_path: str, ckpt: str, sigma_data: float, device: str):
    cfg = expand_env_in_config(yaml.safe_load(open(config_path, encoding="utf-8")))
    cfg["data"]["sigma_data"] = sigma_data
    cfg["data"]["auto_calc_sigma_data"] = False
    cfg["train"]["ckpt_path"] = ""
    model = models.Protpardelle(dict_to_namespace(cfg), device=device).eval()
    state = torch.load(ckpt, weights_only=False, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) else state.state_dict()
    model.load_state_dict(sd)
    return model


def atom_classes(feats: dict) -> dict[str, torch.Tensor]:
    """[N, 37] boolean masks per atom class, all AND-ed with atom_mask."""

    am = feats["atom_mask"] > 0
    noncanon = feats["atom_noncanon"] > 0
    is_hetero = (feats["aatype"] == UNK).unsqueeze(-1) & am
    bb = torch.zeros_like(am)
    bb[:, BB_SLOTS] = True
    out = {
        "bb": am & bb & ~noncanon & ~is_hetero,
        "sc": am & ~bb & ~noncanon & ~is_hetero,
        "ncaa_extra": am & noncanon & ~is_hetero,
        "hetero": is_hetero,
    }
    return out


@torch.no_grad()
def denoise_once(model, feats, sigma, device, use_crop_cond, seed):
    """One denoising pass at noise level `sigma`; returns (pred, clean, masks)."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    coords = feats["atom_positions"].unsqueeze(0).to(device)
    atom_mask = feats["atom_mask"].unsqueeze(0).to(device)
    aatype = feats["aatype"].unsqueeze(0).to(device)
    n = coords.shape[1]
    seq_mask = torch.ones(1, n, device=device)

    crop_cond_mask = None
    struct_crop_cond = None
    if use_crop_cond:
        coords, crop_cond_mask, _ = make_crop_cond_mask_and_recenter_coords(
            atom_mask,
            coords,
            aatype=aatype,
            chain_index=feats["chain_index"].unsqueeze(0).to(device),
            contiguous_prob=0.05,
            discontiguous_prob=0.9,
            sidechain_prob=0.9,
            sidechain_only_prob=0.0,
            max_span_len=12,
            max_discontiguous_res=24,
            dist_threshold=45.0,
            recenter_coords=True,
        )
        struct_crop_cond = coords * crop_cond_mask[..., None]
    else:
        centre = (coords * atom_mask[..., None]).sum((1, 2)) / atom_mask.sum((1, 2))[:, None]
        coords = (coords - centre[:, None, None]) * atom_mask[..., None]

    noise_level = torch.full((1, n), float(sigma), device=device)
    noise = torch.randn(coords.shape, generator=g).to(device)
    noisy = (coords + noise * sigma) * atom_mask[..., None]

    pred, _, _, _ = model(
        noisy_coords=noisy,
        noise_level=noise_level,
        seq_mask=seq_mask,
        residue_index=feats["residue_index"].unsqueeze(0).to(device).long(),
        chain_index=feats["chain_index"].unsqueeze(0).to(device).long(),
        struct_crop_cond=struct_crop_cond,
        crop_cond_mask=crop_cond_mask,
        atom_elem=feats["atom_elem"].unsqueeze(0).to(device),
        atom_ref=feats["atom_ref"].unsqueeze(0).to(device),
        atom_noncanon=feats["atom_noncanon"].unsqueeze(0).to(device),
        run_mpnn_model=False,
    )
    return pred[0].cpu(), coords[0].cpu(), noisy[0].cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    base_dir = "/scratch/users/tianyulu/datasets/pdb_atomized"
    ap.add_argument("--config", default="examples/training/cc152.yaml")
    ap.add_argument("--ckpt", required=True, help="cc152 checkpoint to evaluate")
    ap.add_argument(
        "--base",
        default="/scratch/users/tianyulu/protpardelle-1c-experiments/cc152_init/cc152_from_cc89_epoch415.pth",
        help="cc152-at-init == cc89; the baseline",
    )
    ap.add_argument("--keys", default=f"{base_dir}/val_atomized_pdb_keys.list")
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--ccd-cache", default=f"{base_dir}/ccd_cache.pkl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sigmas", default="0.5,1,2,4,8,16")
    ap.add_argument("--sigma-data", type=float, default=13.5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--require-ncaa", action="store_true", default=True)
    ap.add_argument("--crop-cond", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ccd = atomize_io.load_ccd_cache(args.ccd_cache)
    keys = [k.strip() for k in open(args.keys, encoding="utf-8") if k.strip()]
    keys = sorted(set(keys))

    models_by_name = {
        "cc152": load_model(args.config, args.ckpt, args.sigma_data, args.device),
        "cc89_baseline": load_model(args.config, args.base, args.sigma_data, args.device),
    }
    sigmas = [float(s) for s in args.sigmas.split(",")]

    # sums of squared error and atom counts, per model / sigma / class
    # "noisy" is the reference: RMSD of the un-denoised input (sqrt(3)*sigma for
    # an ideal Gaussian). A model is only useful where it beats this.
    acc: dict = {m: {s: {} for s in sigmas} for m in list(models_by_name) + ["noisy_input"]}
    n_used = 0
    used_keys = []
    for key in keys:
        if n_used >= args.n:
            break
        path = Path(args.mirror) / key[1:3] / f"{key}.cif.gz"
        if not path.exists():
            continue
        feats = atomize_io.load_feats_from_cif_atomized(
            str(path),
            ccd,
            max_tokens=args.max_tokens,
            rng=np.random.default_rng(0),
        )
        if feats is None:
            continue
        cls = atom_classes(feats)
        if args.require_ncaa and cls["ncaa_extra"].sum() == 0:
            continue
        n_used += 1
        used_keys.append(key)
        for mname, model in models_by_name.items():
            for s in sigmas:
                pred, clean, noisy = denoise_once(
                    model, feats, s, args.device, args.crop_cond, seed=n_used
                )
                for tag, arr in ((mname, pred), ("noisy_input", noisy)):
                    if tag == "noisy_input" and mname != list(models_by_name)[0]:
                        continue  # identical across models; count once
                    sq = ((arr - clean) ** 2).sum(-1)
                    for cname, m in cls.items():
                        if m.sum() == 0:
                            continue
                        d = acc[tag][s].setdefault(cname, [0.0, 0])
                        d[0] += float(sq[m].sum())
                        d[1] += int(m.sum())

    print(f"\nevaluated {n_used} structures containing modified residues")
    print(f"crop_cond={args.crop_cond} sigma_data={args.sigma_data}\n")
    classes = ["bb", "sc", "ncaa_extra", "hetero"]
    header = f"{'sigma':>6} {'model':>14} " + " ".join(f"{c:>11}" for c in classes)
    print(header)
    print("-" * len(header))
    results = {}
    for s in sigmas:
        for mname in list(models_by_name) + ["noisy_input"]:
            row = []
            for c in classes:
                d = acc[mname][s].get(c)
                if d is None or d[1] == 0:
                    row.append("        -  ")
                    continue
                rmsd = (d[0] / d[1]) ** 0.5
                results[f"{mname}|{s}|{c}"] = rmsd
                row.append(f"{rmsd:11.3f}")
            print(f"{s:>6} {mname:>14} " + " ".join(row))
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {"rmsd": results, "n_structures": n_used, "keys": used_keys,
                 "crop_cond": args.crop_cond, "ckpt": args.ckpt},
                fh,
                indent=2,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
