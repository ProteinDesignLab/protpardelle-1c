"""Have cc152/cc153 regressed against cc89 on plain canonical protein?

Raw training-loss values are not comparable between cc89 and cc152/cc153: the
Karras weighting is (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2 and
sigma_data is 10.01 for cc89 versus ~13.4 here, the datasets differ (CATH vs
the PDB mirror), and cc153's loss additionally covers ligand atoms. So this
script compares the model-agnostic quantity instead: per-atom-class RMSD of a
single denoising pass at matched absolute noise levels, on the data cc89 was
actually trained for — single-chain, canonical-only, no ligands.

Each model runs at its OWN sigma_data so none is handicapped by being taken
off its training preconditioning.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from protpardelle.common import residue_constants as rc
from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config

BB = [rc.atom_order[a] for a in ("N", "CA", "C", "O")]


def build(cfg_path, ckpt, sigma_data, device):
    d = expand_env_in_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    d["data"]["sigma_data"] = sigma_data
    d["data"]["auto_calc_sigma_data"] = False
    d["train"]["ckpt_path"] = ""
    m = models.Protpardelle(dict_to_namespace(d), device=device).eval()
    c = torch.load(ckpt, weights_only=False, map_location=device)
    m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c.state_dict())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/scratch/users/tianyulu/datasets/pdb_atomized"
    exp = "/scratch/users/tianyulu/protpardelle-1c-experiments"
    ap.add_argument("--epoch", type=int, default=44)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--sigmas", default="0.5,1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    entries = [
        ("cc89",  "examples/training/cc152.yaml",
         f"{exp}/cc152_init/cc152_from_cc89_epoch415.pth", 10.01),
        ("cc152", "examples/training/cc152.yaml",
         f"{exp}/protpardelle-1c/cc152/checkpoints/epoch{a.epoch}_training_state.pth", 13.35),
        ("cc153", "examples/training/cc153.yaml",
         f"{exp}/protpardelle-1c/cc153/checkpoints/epoch{a.epoch}_training_state.pth", 13.41),
    ]
    nets = {n: build(c, k, s, a.device) for n, c, k, s in entries}
    sigmas = [float(x) for x in a.sigmas.split(",")]

    ccd = A.maybe_load_default_ccd()
    mirror = "/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF"
    keys = sorted(set(k.strip() for k in
                      open(f"{base}/val_atomized_pdb_keys.list", encoding="utf-8") if k.strip()))

    # cc89's home turf: single chain, canonical residues only, no hetero.
    picks = []
    for k in keys:
        if len(picks) >= a.n:
            break
        p = f"{mirror}/{k[1:3]}/{k}.cif.gz"
        try:
            f = A.load_feats_from_cif_atomized(
                p, ccd, max_tokens=a.max_tokens, include_hetero=False,
                rng=np.random.default_rng(0))
        except Exception:
            continue
        if f is None or f["n_modified"] > 0:
            continue
        if int(f["chain_index"].max()) != 0:
            continue
        if f["atom_mask"][:, A.N_ATOM37:].sum() > 0:
            continue
        picks.append((k, f))
    print(f"{len(picks)} plain single-chain canonical proteins: "
          f"{[k for k, _ in picks]}\n")

    acc = {n: {s: {"bb": [0.0, 0], "sc": [0.0, 0]} for s in sigmas} for n in nets}
    acc["noisy"] = {s: {"bb": [0.0, 0], "sc": [0.0, 0]} for s in sigmas}

    for si, (key, f) in enumerate(picks):
        am = f["atom_mask"] > 0
        bbm = torch.zeros_like(am)
        bbm[:, BB] = True
        cls = {"bb": am & bbm, "sc": am & ~bbm}
        coords0 = f["atom_positions"].unsqueeze(0).to(a.device)
        amk = f["atom_mask"].unsqueeze(0).to(a.device)
        n = coords0.shape[1]
        np.random.seed(si)
        torch.manual_seed(si)
        coords, ccm, _ = make_crop_cond_mask_and_recenter_coords(
            amk, coords0, aatype=f["aatype"].unsqueeze(0).to(a.device),
            chain_index=f["chain_index"].unsqueeze(0).to(a.device),
            contiguous_prob=0.05, discontiguous_prob=0.9, sidechain_prob=0.9,
            sidechain_only_prob=0.0, max_span_len=12, max_discontiguous_res=24,
            dist_threshold=45.0, recenter_coords=True)
        for sg in sigmas:
            g = torch.Generator().manual_seed(si)
            noisy = (coords + torch.randn(coords.shape, generator=g).to(a.device) * sg) \
                * amk[..., None]
            for cn, m in cls.items():
                d = acc["noisy"][sg][cn]
                d[0] += float(((noisy[0].cpu() - coords[0].cpu()) ** 2).sum(-1)[m].sum())
                d[1] += int(m.sum())
            for name, net in nets.items():
                with torch.no_grad():
                    pred = net(
                        noisy_coords=noisy,
                        noise_level=torch.full((1, n), sg, device=a.device),
                        seq_mask=torch.ones(1, n, device=a.device),
                        residue_index=f["residue_index"].unsqueeze(0).to(a.device).long(),
                        chain_index=f["chain_index"].unsqueeze(0).to(a.device).long(),
                        struct_crop_cond=coords * ccm[..., None], crop_cond_mask=ccm,
                        atom_elem=f["atom_elem"].unsqueeze(0).to(a.device),
                        atom_ref=f["atom_ref"].unsqueeze(0).to(a.device),
                        atom_noncanon=f["atom_noncanon"].unsqueeze(0).to(a.device),
                        run_mpnn_model=False)[0]
                sq = ((pred[0].cpu() - coords[0].cpu()) ** 2).sum(-1)
                for cn, m in cls.items():
                    d = acc[name][sg][cn]
                    d[0] += float(sq[m].sum())
                    d[1] += int(m.sum())

    print(f"{'sigma':>6} {'model':>8} {'backbone':>10} {'sidechain':>11}")
    print("-" * 40)
    for sg in sigmas:
        for name in list(nets) + ["noisy"]:
            r = []
            for cn in ("bb", "sc"):
                d = acc[name][sg][cn]
                r.append((d[0] / d[1]) ** 0.5 if d[1] else float("nan"))
            print(f"{sg:>6} {name:>8} {r[0]:10.3f} {r[1]:11.3f}")
        print()


if __name__ == "__main__":
    main()
