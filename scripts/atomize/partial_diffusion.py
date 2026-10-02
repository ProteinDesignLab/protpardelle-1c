"""Partial diffusion on an atomized structure: rewind N steps, then denoise.

This is the Sampleworks use case — take an experimental structure, push it back
up the noise schedule, and denoise it several times to get an ensemble of
plausible neighbouring conformations.

The integrator is the same EDM Euler step the main sampler uses:

    score = (x_t - x0_hat) / sigma_t
    x_next = x_t + step_scale * score * (sigma_next - sigma_t)

with self-conditioning carried between steps, exactly as in
``Protpardelle.sample``. The noise schedule is the model's own sampling
schedule (uniform/Karras, t=1 high noise -> t=0 low noise).

Control the rewind with ``--sigma-start``, not a step count. The schedule is
steeply Karras-shaped and ``s_max: 80`` is in units of sigma_data, so sigma
runs from 800 down to 0.01 over the 200 steps: index 50 is still sigma=167,
~12x the data std, i.e. full de novo generation. Useful partial diffusion
lives at sigma of order 1-10 against a sigma_data of ~13.

Ligand handling follows the model's own `ligand_treatment`:
  * cc152 (conditioning_only) — ligand atoms are held at their crystal
    coordinates throughout, which is what a caller fitting protein ensembles
    around a fixed cofactor wants.
  * cc153 (denoised) — ligands are noised and denoised with everything else.

Usage:
    python scripts/atomize/partial_diffusion.py --pdb-id 7z0e \
        --ckpt <...>/cc153/checkpoints/epoch41_training_state.pth \
        --config examples/training/cc153.yaml \
        --n-steps 200 --rewind 150 --num-samples 8 --out-dir <...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from protpardelle.core import diffusion, models
from protpardelle.data import atomize_io as A
from protpardelle.utils import dict_to_namespace, expand_env_in_config


def build(cfg_path: str, ckpt: str, sigma_data: float, device: str):
    cfg = expand_env_in_config(yaml.safe_load(open(cfg_path, encoding="utf-8")))
    cfg["data"]["sigma_data"] = sigma_data
    cfg["data"]["auto_calc_sigma_data"] = False
    cfg["train"]["ckpt_path"] = ""
    model = models.Protpardelle(dict_to_namespace(cfg), device=device).eval()
    state = torch.load(ckpt, weights_only=False, map_location=device)
    sd = state["model_state_dict"] if isinstance(state, dict) else state.state_dict()
    model.load_state_dict(sd)
    return model, cfg


@torch.no_grad()
def partial_diffuse(
    model, feats, *, n_steps, rewind, sigma_data, step_scale, clamp_ligand,
    device, seed, sigma_start_req=None,
):
    """Noise to sigmas[n_steps-rewind], then run the remaining Euler steps."""

    g = torch.Generator(device="cpu").manual_seed(seed)
    coords0 = feats["atom_positions"].unsqueeze(0).to(device)
    atom_mask = feats["atom_mask"].unsqueeze(0).to(device)
    n = coords0.shape[1]
    seq_mask = torch.ones(1, n, device=device)
    lig_atom = (
        (feats["is_ligand"].unsqueeze(-1).to(device) > 0) & (atom_mask[0] > 0)
    ).unsqueeze(0)

    # centre on the structure, as training does
    centre = (coords0 * atom_mask[..., None]).sum((1, 2)) / atom_mask.sum((1, 2))[:, None]
    coords0 = (coords0 - centre[:, None, None]) * atom_mask[..., None]

    # the model's own sampling schedule; t = 1 is high noise
    ts = torch.linspace(1, 0, n_steps + 1, device=device)
    sigmas = model.sampling_noise_schedule_default(ts)
    if sigma_start_req is not None:
        # Enter the schedule at the step whose sigma is closest to the request.
        # This is the meaningful control: the schedule is steeply Karras-shaped
        # (s_max=80 is in units of sigma_data, so sigma runs 800 -> 0.01), and a
        # step count alone is a very poor proxy for how much structure survives.
        start = int(torch.argmin((sigmas - sigma_start_req).abs()).item())
        start = min(start, n_steps - 1)
    else:
        start = n_steps - rewind
    sigma_start = float(sigmas[start])

    noise = torch.randn(coords0.shape, generator=g).to(device)
    xt = (coords0 + noise * sigma_start) * atom_mask[..., None]
    if clamp_ligand:
        xt = torch.where(lig_atom.unsqueeze(-1), coords0, xt)

    struct_self_cond = None
    kw = dict(
        seq_mask=seq_mask,
        residue_index=feats["residue_index"].unsqueeze(0).to(device).long(),
        chain_index=feats["chain_index"].unsqueeze(0).to(device).long(),
        atom_elem=feats["atom_elem"].unsqueeze(0).to(device),
        atom_ref=feats["atom_ref"].unsqueeze(0).to(device),
        atom_noncanon=feats["atom_noncanon"].unsqueeze(0).to(device),
        run_mpnn_model=False,
    )

    for i in range(start, n_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        nl = torch.full((1, n), float(s_cur), device=device)
        x0_hat, _, struct_self_cond, _ = model(
            noisy_coords=xt, noise_level=nl, struct_self_cond=struct_self_cond, **kw
        )
        score = (xt - x0_hat) / max(float(s_cur), 1e-6)
        xt = xt + step_scale * score * float(s_next - s_cur)
        xt = xt * atom_mask[..., None]
        if clamp_ligand:
            xt = torch.where(lig_atom.unsqueeze(-1), coords0, xt)

    return xt[0].cpu(), coords0[0].cpu(), sigma_start


def rmsd(a, b, mask):
    return float((((a - b) ** 2).sum(-1)[mask]).mean() ** 0.5)


def bond_dev(xyz, feats, rows):
    """Covalent-bond integrity: mean |d_obs - d_ideal| and %% of bonds >0.3 A off.

    Distances are compared against the CCD ideal conformer carried in
    ``atom_ref``, restricted to pairs whose ideal separation is < 1.8 A, i.e.
    actual covalent bonds. Including pairs across a rotatable bond would just
    measure torsional freedom (a crystal structure scores ~0.4 A on that).
    A ligand token duplicates its anchor atom into slot 1; that duplicate is
    excluded or it reads as an atom bonded to itself.
    """

    ref = feats["atom_ref"].numpy() * A.REF_COORD_SCALE
    mask = feats["atom_mask"].numpy() > 0
    has_ref = (np.abs(ref).sum(-1) > 0) & mask
    is_lig = feats["is_ligand"].numpy() > 0
    xyz = xyz.numpy() if hasattr(xyz, "numpy") else xyz

    devs = []
    for i in rows:
        sel = np.where(has_ref[i])[0]
        if is_lig[i]:
            sel = sel[sel != A.ANCHOR_SLOT]
        if len(sel) < 3:
            continue
        o, r = xyz[i][sel], ref[i][sel]
        do = np.linalg.norm(o[:, None] - o[None], axis=-1)
        dr = np.linalg.norm(r[:, None] - r[None], axis=-1)
        iu = np.triu_indices(len(sel), 1)
        bonded = dr[iu] < 1.8
        if bonded.sum():
            devs.append(np.abs(do[iu] - dr[iu])[bonded])
    if not devs:
        return float("nan"), float("nan")
    d = np.concatenate(devs)
    return float(d.mean()), float(100 * (d > 0.3).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/scratch/users/tianyulu/datasets/pdb_atomized"
    ap.add_argument("--pdb-ids", default="7z0e",
                    help="comma-separated PDB ids, or @file with one per line")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sigma-data", type=float, required=True)
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--ccd-cache", default=f"{base}/ccd_cache.pkl")
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--rewind", type=int, default=150)
    ap.add_argument("--sigma-start", type=float, default=None,
                    help="Enter the schedule at this noise level (overrides --rewind). "
                         "This, not the step count, controls how much structure survives.")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--step-scale", type=float, default=1.0)
    # NOT the training default of 384: the featurizer crops to the *nearest*
    # max_tokens residues around an anchor, which is right for training but
    # fragments the chain for sampling (a 617-residue protein cropped to 384
    # came out with 31 spurious backbone breaks). Default large enough that
    # nothing is cropped, and warn if anything still is.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    label = a.label or Path(a.config).stem
    out_root = Path(a.out_dir) / label
    out_root.mkdir(parents=True, exist_ok=True)

    ids = a.pdb_ids
    if ids.startswith("@"):
        ids = ",".join(l.strip() for l in open(ids[1:], encoding="utf-8") if l.strip())
    pdb_ids = [x.strip().lower() for x in ids.split(",") if x.strip()]

    ccd = A.load_ccd_cache(a.ccd_cache)
    model, cfg = build(a.config, a.ckpt, a.sigma_data, a.device)
    clamp = cfg["model"].get("ligand_treatment") == "conditioning_only"
    print(f"{label}: {len(pdb_ids)} structures, {a.num_samples} samples each, "
          f"clamp_ligand={clamp}, sigma_data={a.sigma_data}\n")

    hdr = (f"{'pdb':>6} {'tok':>5} {'ncaa':>5} {'lig':>5} | {'bb':>5} {'sc':>5} "
           f"{'ncaa':>6} {'lig':>6} | {'spread':>6} | {'bondDev':>7} {'%>0.3':>6} "
           f"{'ligBond':>8} {'%>0.3':>6}")
    print(hdr); print("-" * len(hdr))

    allrows = []
    for key in pdb_ids:
        path = Path(a.mirror) / key[1:3] / f"{key}.cif.gz"
        if not path.exists():
            print(f"{key:>6}  MISSING {path}"); continue
        try:
            feats = A.load_feats_from_cif_atomized(
                str(path), ccd, max_tokens=a.max_tokens, rng=np.random.default_rng(0))
        except Exception as e:
            print(f"{key:>6}  FAILED {type(e).__name__}: {e}"); continue
        if feats is None:
            print(f"{key:>6}  FAILED to featurize"); continue
        n_tok = int(feats["atom_positions"].shape[0])
        if n_tok >= a.max_tokens:
            print(f"{key:>6}  WARNING cropped to {a.max_tokens} tokens -- the crop is "
                  f"spatial, so expect spurious backbone breaks. Raise --max-tokens.")

        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        am = feats["atom_mask"] > 0
        lig = (feats["is_ligand"].unsqueeze(-1) > 0) & am
        nc = (feats["atom_noncanon"] > 0) & am & ~lig
        prot = am & ~lig
        bb = torch.zeros_like(am)
        for sl in (0, 1, 2, 4):
            bb[:, sl] = True
        cls = {"bb": prot & bb, "sc": prot & ~bb & ~nc, "ncaa": nc, "ligand": lig}
        prot_rows = [i for i in range(len(feats["is_ligand"])) if feats["is_ligand"][i] <= 0]
        lig_rows = [i for i in range(len(feats["is_ligand"])) if feats["is_ligand"][i] > 0]

        (out / f"{key}_input.pdb").write_text(A.feats_to_pdb_str_atomized(feats))

        rows, samples, bd, bdp, lbd, lbdp = [], [], [], [], [], []
        for i in range(a.num_samples):
            xt, coords0, sigma_start = partial_diffuse(
                model, feats, n_steps=a.n_steps, rewind=a.rewind,
                sigma_data=a.sigma_data, step_scale=a.step_scale,
                clamp_ligand=clamp, device=a.device, seed=1000 + i,
                sigma_start_req=a.sigma_start)
            samples.append(xt)
            row = {k: rmsd(xt, coords0, m) for k, m in cls.items() if m.any()}
            rows.append(row)
            m_, p_ = bond_dev(xt, feats, prot_rows)
            bd.append(m_); bdp.append(p_)
            if lig_rows:
                ml, pl = bond_dev(xt, feats, lig_rows)
                lbd.append(ml); lbdp.append(pl)
            (out / f"{key}_sample{i}.pdb").write_text(
                A.feats_to_pdb_str_atomized(feats, coords=xt))

        pair = [rmsd(samples[i], samples[j], am)
                for i in range(len(samples)) for j in range(i + 1, len(samples))]
        g = lambda k: np.mean([r[k] for r in rows]) if k in rows[0] else float("nan")
        rec = dict(pdb=key, tokens=int(feats["atom_positions"].shape[0]),
                   n_ncaa=int(nc.sum()), n_lig=int(lig.sum()),
                   bb=g("bb"), sc=g("sc"), ncaa=g("ncaa"), ligand=g("ligand"),
                   spread=float(np.mean(pair)),
                   bond_dev=float(np.mean(bd)), bond_pct=float(np.mean(bdp)),
                   lig_bond_dev=float(np.mean(lbd)) if lbd else float("nan"),
                   lig_bond_pct=float(np.mean(lbdp)) if lbdp else float("nan"),
                   sigma_start=sigma_start)
        allrows.append(rec)
        print(f"{key:>6} {rec['tokens']:5d} {rec['n_ncaa']:5d} {rec['n_lig']:5d} | "
              f"{rec['bb']:5.2f} {rec['sc']:5.2f} {rec['ncaa']:6.2f} {rec['ligand']:6.2f} | "
              f"{rec['spread']:6.2f} | {rec['bond_dev']:7.3f} {rec['bond_pct']:6.1f} "
              f"{rec['lig_bond_dev']:8.3f} {rec['lig_bond_pct']:6.1f}")

        json.dump({**rec, "per_sample_rmsd_to_input": rows, "pairwise_rmsd": pair,
                   "ckpt": a.ckpt, "clamp_ligand": clamp},
                  open(out / "summary.json", "w"), indent=2)

    if allrows:
        def mean(k):
            v = [r[k] for r in allrows if not np.isnan(r[k])]
            return np.mean(v) if v else float("nan")
        print("-" * len(hdr))
        print(f"{'MEAN':>6} {'':5} {'':5} {'':5} | {mean('bb'):5.2f} {mean('sc'):5.2f} "
              f"{mean('ncaa'):6.2f} {mean('ligand'):6.2f} | {mean('spread'):6.2f} | "
              f"{mean('bond_dev'):7.3f} {mean('bond_pct'):6.1f} "
              f"{mean('lig_bond_dev'):8.3f} {mean('lig_bond_pct'):6.1f}")
        json.dump(allrows, open(out_root / "all_structures.json", "w"), indent=2)
        print(f"\nwrote {len(allrows)} structures to {out_root}")


if __name__ == "__main__":
    main()
