"""cc152 (ligands as context) vs cc153 (ligands denoised).

The two models answer different questions, so a single RMSD column is not
enough:

* **cc153** noises and supervises ligand atoms, so "can it place the ligand?"
  is meaningful. Reported as the `ligand` column.
* **cc152** is handed the ligand clean and never predicts it, so its ligand
  column is trivially ~0. The question there is whether having the ligand as
  context makes the *protein around it* better. That is the `pocket` column:
  protein atoms within `--pocket-radius` of any ligand atom.

The ablation is the same idea applied by deletion: run each model with the
ligand tokens **removed from the input entirely** and see how much the pocket
degrades. If conditioning on ligand coordinates is doing real work, dropping
them should hurt the pocket and leave the rest of the protein alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from protpardelle.common import residue_constants as rc
from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config

BB_SLOTS = [rc.atom_order[a] for a in ("N", "CA", "C", "O")]


def build(cfg_path: str, ckpt: str, sigma: float, device: str):
    d = expand_env_in_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    d["data"]["sigma_data"] = sigma
    d["data"]["auto_calc_sigma_data"] = False
    d["train"]["ckpt_path"] = ""
    m = models.Protpardelle(dict_to_namespace(d), device=device).eval()
    c = torch.load(ckpt, weights_only=False, map_location=device)
    m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c.state_dict())
    return m


def drop_ligands(feats: dict) -> dict:
    """Same structure with every ligand/metal token removed."""

    keep = feats["is_ligand"] <= 0
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == keep.shape[0]:
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == keep.shape[0]:
            out[k] = [x for x, m in zip(v, keep.tolist()) if m]
        else:
            out[k] = v
    return out


def prepare(feats, sigma, device, seed):
    """Recentred coords, crop-cond mask and noise for the FULL structure.

    Computed once and then *indexed* for the ligand-dropped variant, so every
    variant sees the same motif, the same recentring and the same noise on the
    atoms they share. Re-drawing these per variant makes the comparison
    meaningless (a small motif draw alone can cost several Angstrom).
    """

    coords = feats["atom_positions"].unsqueeze(0).to(device)
    atom_mask = feats["atom_mask"].unsqueeze(0).to(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    coords, ccm, _ = make_crop_cond_mask_and_recenter_coords(
        atom_mask, coords,
        aatype=feats["aatype"].unsqueeze(0).to(device),
        chain_index=feats["chain_index"].unsqueeze(0).to(device),
        contiguous_prob=0.05, discontiguous_prob=0.9, sidechain_prob=0.9,
        sidechain_only_prob=0.0, max_span_len=12, max_discontiguous_res=24,
        dist_threshold=45.0, recenter_coords=True,
    )
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(coords.shape, generator=g).to(device) * sigma
    return coords, ccm, noise


@torch.no_grad()
def run(model, feats, coords, ccm, noise, sigma, device, clean_ligands):
    atom_mask = feats["atom_mask"].unsqueeze(0).to(device)
    n = coords.shape[1]
    lig_atom = (
        (feats["is_ligand"].unsqueeze(-1).to(device) > 0) & (feats["atom_mask"].to(device) > 0)
    ).unsqueeze(0)

    if clean_ligands:
        ccm = torch.clamp(ccm + lig_atom.to(ccm.dtype), max=1.0)
    scc = coords * ccm[..., None]
    noisy = (coords + noise) * atom_mask[..., None]
    if clean_ligands:
        noisy = torch.where(lig_atom.unsqueeze(-1), coords, noisy)

    pred = model(
        noisy_coords=noisy,
        noise_level=torch.full((1, n), float(sigma), device=device),
        seq_mask=torch.ones(1, n, device=device),
        residue_index=feats["residue_index"].unsqueeze(0).to(device).long(),
        chain_index=feats["chain_index"].unsqueeze(0).to(device).long(),
        struct_crop_cond=scc, crop_cond_mask=ccm,
        atom_elem=feats["atom_elem"].unsqueeze(0).to(device),
        atom_ref=feats["atom_ref"].unsqueeze(0).to(device),
        atom_noncanon=feats["atom_noncanon"].unsqueeze(0).to(device),
        run_mpnn_model=False,
    )[0]
    return pred[0].cpu(), coords[0].cpu()


def classes(feats: dict, pocket_radius: float) -> dict:
    am = feats["atom_mask"] > 0
    lig = (feats["is_ligand"].unsqueeze(-1) > 0) & am
    prot = am & ~lig
    nc = (feats["atom_noncanon"] > 0) & prot
    bb = torch.zeros_like(am)
    bb[:, BB_SLOTS] = True

    pocket = torch.zeros_like(am)
    if lig.any():
        lx = feats["atom_positions"][lig]
        px = feats["atom_positions"]
        d = torch.cdist(px.reshape(-1, 3), lx).min(-1).values.reshape(px.shape[:2])
        pocket = prot & (d <= pocket_radius)
    return {
        "bb": prot & bb & ~nc,
        "sc": prot & ~bb & ~nc,
        "ncaa": nc,
        "pocket": pocket,
        "ligand": lig,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/scratch/users/tianyulu/datasets/pdb_atomized"
    ap.add_argument("--cc152-ckpt", required=True)
    ap.add_argument("--cc153-ckpt", required=True)
    ap.add_argument("--cc152-sigma-data", type=float, default=13.23)
    ap.add_argument("--cc153-sigma-data", type=float, default=13.61)
    ap.add_argument("--keys", default=f"{base}/val_atomized_pdb_keys.list")
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--ccd-cache", default=f"{base}/ccd_cache.pkl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--sigmas", default="1,4,8")
    ap.add_argument("--pocket-radius", type=float, default=6.0)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ccd = A.load_ccd_cache(a.ccd_cache)
    keys = sorted(set(k.strip() for k in open(a.keys, encoding="utf-8") if k.strip()))
    sigmas = [float(s) for s in a.sigmas.split(",")]

    # (model, clean_ligands, drop_ligands)
    variants = [
        ("cc152 (ligand ctx)", "cc152", True, False),
        ("cc152 no-ligand", "cc152", True, True),
        ("cc153 (denoised)", "cc153", False, False),
        ("cc153 no-ligand", "cc153", False, True),
    ]
    nets = {
        "cc152": build("examples/training/cc152.yaml", a.cc152_ckpt, a.cc152_sigma_data, a.device),
        "cc153": build("examples/training/cc153.yaml", a.cc153_ckpt, a.cc153_sigma_data, a.device),
    }
    CLS = ["bb", "sc", "ncaa", "pocket", "ligand"]
    acc = {s: {v[0]: {c: [0.0, 0] for c in CLS} for v in variants} for s in sigmas}

    used = 0
    for key in keys:
        if used >= a.n:
            break
        p = Path(a.mirror) / key[1:3] / f"{key}.cif.gz"
        if not p.exists():
            continue
        feats = A.load_feats_from_cif_atomized(
            str(p), ccd, max_tokens=a.max_tokens, rng=np.random.default_rng(0))
        if feats is None or feats["is_ligand"].sum() == 0:
            continue  # only structures that actually have a ligand
        used += 1
        full_cls = classes(feats, a.pocket_radius)
        dropped = drop_ligands(feats)
        drop_cls = classes(dropped, a.pocket_radius)
        # "pocket" must mean the same atoms whether or not the ligand token is
        # present, so carry the full-structure definition over to the dropped
        # variant rather than recomputing it (there is no ligand to measure from).
        drop_cls["pocket"] = full_cls["pocket"][feats["is_ligand"] <= 0]

        keepmask = (feats["is_ligand"] <= 0).to(a.device)
        for sigma in sigmas:
            coords_f, ccm_f, noise_f = prepare(feats, sigma, a.device, seed=1000 + used)
            # index the SAME tensors for the dropped variant
            coords_d = coords_f[:, keepmask]
            ccm_d = ccm_f[:, keepmask]
            noise_d = noise_f[:, keepmask]
            for label, net, clean, drop in variants:
                f = dropped if drop else feats
                cls = drop_cls if drop else full_cls
                c_, m_, z_ = (coords_d, ccm_d, noise_d) if drop else (coords_f, ccm_f, noise_f)
                pred, clean_xyz = run(nets[net], f, c_, m_, z_, sigma, a.device, clean)
                sq = ((pred - clean_xyz) ** 2).sum(-1)
                for cn in CLS:
                    m = cls.get(cn)
                    if m is None or m.sum() == 0:
                        continue
                    d = acc[sigma][label][cn]
                    d[0] += float(sq[m].sum())
                    d[1] += int(m.sum())

    print(f"\n{used} held-out structures containing a ligand/metal; "
          f"pocket = protein atoms within {a.pocket_radius} A of a ligand atom\n")
    hdr = f"{'sigma':>5} {'variant':>20} " + " ".join(f"{c:>9}" for c in CLS)
    print(hdr); print("-" * len(hdr))
    out = {}
    for sigma in sigmas:
        for label, *_ in variants:
            row = []
            for cn in CLS:
                d = acc[sigma][label][cn]
                if d[1] == 0:
                    row.append("      -  "); continue
                v = (d[0] / d[1]) ** 0.5
                out[f"{label}|{sigma}|{cn}"] = v
                row.append(f"{v:9.3f}")
            print(f"{sigma:>5} {label:>20} " + " ".join(row))
        print()
    if a.out:
        json.dump({"rmsd": out, "n": used}, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
