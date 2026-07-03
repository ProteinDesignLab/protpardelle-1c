"""Genie3 binder-design benchmark: sequence design + AF2-Multimer eval.

Adapted from ``scripts/bindcraft/af2.py`` for the Genie3 binder benchmark
(``examples/sampling/genie3_benchmark.yaml``). Follows the genie3 AF2M protocol:
multimer weights, target structure used as template + initial guess, with the
genie3 selection criteria.

Selection criteria (Genie3 paper):
- Binder pTM > 0.8 (interface pTM in the multimer prediction)
- Min interaction pAE < 1.5 Å (min across binder x target residue pairs)
- Complex RMSD < 2.5 Å (predicted vs designed CA, target-aligned)
- Interface coverage: >= 80% of user-specified hotspot residues within 5 Å of
  the binder when interface has >= 4 residues; 100% if <= 3 residues
- Distances computed over closest heavy atoms

Sequence design is done with ProteinMPNN (LigandMPNN repo, protein_mpnn mode)
and optionally with ensemble-conditioned Caliby. Both run in their own uv envs
via subprocess.

Run inside the colabdesign uv env::

    /scratch/users/tianyulu/envs/colabdesign/bin/python scripts/genie3/af2.py \\
        --samples-dir /scratch/users/tianyulu/protpardelle-1c-sampling-results/genie3_benchmark/cc141-... \\
        --output-dir <out> \\
        --seq-design proteinmpnn caliby \\
        --num-samples-per-motif 1 --num-seqs 2 --num-models 1 --num-recycles 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -----------------------------------------------------------------------------
# Paths (override via CLI flags or env vars)
# -----------------------------------------------------------------------------
GENIE3_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "sampling" / "genie3_benchmark.yaml"
GENIE3_TARGET_DIR = Path("/home/users/tianyulu/code/genie3/data/design/binder_design/binderbench/targets/pdb")
AF2_PARAM_DIR = "/oak/stanford/groups/possu/tianyu/models/af2/params"
PROTEINMPNN_WEIGHTS = "/oak/stanford/groups/possu/tianyu/models/ProteinMPNN/v_48_020.pt"
LIGANDMPNN_REPO = "/home/users/tianyulu/code/LigandMPNN"
LIGANDMPNN_PY = "/scratch/users/tianyulu/envs/ligandmpnn/bin/python"
CALIBY_PY = "/scratch/users/tianyulu/envs/caliby/bin/python"

_SCRIPT_DIR = Path(__file__).resolve().parent

# Genie3 thresholds
GENIE3_PTM_MIN = 0.8
GENIE3_IPAE_MAX = 1.5  # Angstroms
GENIE3_RMSD_MAX = 2.5  # Angstroms
GENIE3_INTERFACE_DIST_A = 5.0
GENIE3_INTERFACE_FRAC = 0.8  # 80% for >=4 hotspots, 100% otherwise

# -----------------------------------------------------------------------------
# Genie3 yaml + sample directory parsing
# -----------------------------------------------------------------------------


def load_genie3_config(yaml_path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    motifs = cfg["motifs"]
    contigs = cfg["motif_contigs"]
    hotspots = cfg.get("hotspots", [None] * len(motifs))
    info = {}
    for i, m in enumerate(motifs):
        if isinstance(m, list):
            m_name = m[0][0] if isinstance(m[0], (list, tuple)) else m[0]
        else:
            m_name = m
        info[m_name] = {
            "contig": contigs[i],
            "hotspots": hotspots[i],
        }
    return info


def parse_contig_target_chains(contig):
    """Return the original target chain IDs in contig order.

    Contig example: ``"B1-322;/;C1-170;/;80-120"`` → ``["B", "C"]``.
    The trailing scaffold (binder) segment has no chain letter.
    """
    chains = []
    for seg in contig.split(";"):
        seg = seg.strip()
        if not seg or seg == "/":
            continue
        if seg[0].isalpha():
            ch = seg[0]
            if ch not in chains:
                chains.append(ch)
    return chains


def chain_residue_counts(pdb_path):
    """Return ``OrderedDict`` of chain_id → sorted list of residue numbers."""
    chains: OrderedDict[str, list[int]] = OrderedDict()
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            resnum = int(line[22:26])
            key = (ch, resnum)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(ch, []).append(resnum)
    return chains


def split_sample_chains(sample_pdb, target_orig_chains):
    """Identify target vs binder chains in a sample PDB.

    Sample chains are alphabetically reassigned: the first N_target chains
    correspond to ``target_orig_chains`` (in contig order). The remaining chain
    is the binder.

    Returns ``(target_chains_in_sample, binder_chain_in_sample,
              orig_to_sample_map, chain_residue_counts_dict)``.
    """
    chains = chain_residue_counts(sample_pdb)
    chain_ids = list(chains.keys())
    n_target = len(target_orig_chains)
    if len(chain_ids) < n_target + 1:
        raise RuntimeError(
            f"{sample_pdb}: expected at least {n_target + 1} chains, got {chain_ids}"
        )
    target_chains = chain_ids[:n_target]
    binder_chain = chain_ids[n_target]
    orig_to_sample = dict(zip(target_orig_chains, target_chains))
    return target_chains, binder_chain, orig_to_sample, chains


def parse_hotspot_tokens(hotspot_str):
    """Parse the yaml hotspot string into a list of ``(orig_chain, resnum)`` pairs."""
    if not hotspot_str:
        return []
    pairs = []
    for tok in hotspot_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        pairs.append((tok[0], int(tok[1:])))
    return pairs


def parse_hotspots(hotspot_str, orig_to_sample):
    """Convert genie3 hotspot string (orig chain IDs) → list of (sample_chain, resnum)."""
    return [(orig_to_sample.get(c, c), r) for c, r in parse_hotspot_tokens(hotspot_str)]


def map_hotspots_to_predicted(hotspot_str, target_chain_lengths):
    """Map genie3 hotspots (orig chain IDs) to (chain_A, resnum) in the colabdesign
    predicted PDB, which merges all target chains into chain A with a +50 gap.

    Args:
        hotspot_str: e.g. ``"B64,C45"``
        target_chain_lengths: ordered mapping ``{orig_chain: n_residues}`` covering
            every target chain in contig order.

    Returns: list of ``(chain="A", resnum_in_predicted)``.
    """
    if not hotspot_str:
        return []
    offsets = {}
    last = 0
    for ch, n in target_chain_lengths.items():
        offsets[ch] = last
        last = last + n + 50
    out = []
    for orig_ch, orig_res in parse_hotspot_tokens(hotspot_str):
        if orig_ch not in offsets:
            continue
        out.append(("A", offsets[orig_ch] + orig_res))
    return out


def target_chain_lengths_from_target_pdb(target_pdb, target_orig_chains):
    """Return an ``OrderedDict {chain: n_residues}`` for each target chain in order."""
    chains = chain_residue_counts(target_pdb)
    return OrderedDict((c, len(chains[c])) for c in target_orig_chains)


def build_fixed_pos_seq(target_chains_in_sample, chains_dict):
    """Build a Caliby ``fixed_pos_seq`` string covering all target residues.

    Example: ``"A1-157,B523-692"``.
    """
    parts = []
    for ch in target_chains_in_sample:
        resnums = sorted(chains_dict[ch])
        # Compress to ranges
        if not resnums:
            continue
        start = prev = resnums[0]
        for r in resnums[1:]:
            if r == prev + 1:
                prev = r
                continue
            parts.append(f"{ch}{start}-{prev}" if start != prev else f"{ch}{start}")
            start = prev = r
        parts.append(f"{ch}{start}-{prev}" if start != prev else f"{ch}{start}")
    return ",".join(parts)


# -----------------------------------------------------------------------------
# Sequence design dispatch (subprocess to other envs)
# -----------------------------------------------------------------------------


def run_proteinmpnn(samples, num_seqs, temperature, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "samples": samples,
        "num_seqs": num_seqs,
        "temperature": temperature,
        "seed": seed,
        "out_dir": str(out_dir),
        "ligandmpnn_repo": LIGANDMPNN_REPO,
        "checkpoint": PROTEINMPNN_WEIGHTS,
    }
    jobs_path = out_dir / "_pmpnn_jobs.json"
    res_path = out_dir / "_pmpnn_results.json"
    with open(jobs_path, "w") as f:
        json.dump(spec, f, indent=2)
    cmd = [LIGANDMPNN_PY, str(_SCRIPT_DIR / "_pmpnn_runner.py"), "--jobs", str(jobs_path), "--out", str(res_path)]
    subprocess.run(cmd, check=True)
    with open(res_path) as f:
        return json.load(f)["results"]


def run_caliby(samples, num_seqs, num_conformers, use_ensemble, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "samples": samples,
        "num_seqs": num_seqs,
        "num_conformers": num_conformers,
        "use_ensemble": use_ensemble,
        "design_out_dir": str(out_dir / "design"),
        "ensemble_out_dir": str(out_dir / "ensembles"),
    }
    jobs_path = out_dir / "_caliby_jobs.json"
    res_path = out_dir / "_caliby_results.json"
    with open(jobs_path, "w") as f:
        json.dump(spec, f, indent=2)
    cmd = [CALIBY_PY, str(_SCRIPT_DIR / "_caliby_runner.py"), "--jobs", str(jobs_path), "--out", str(res_path)]
    subprocess.run(cmd, check=True)
    with open(res_path) as f:
        return json.load(f)["results"]


# -----------------------------------------------------------------------------
# Heavy-atom interface coverage on a predicted complex PDB
# -----------------------------------------------------------------------------


def _atoms_per_chain(pdb_path):
    """Return ``dict chain → list[(resnum, atom_name, xyz)]`` for ATOM records.

    Hydrogens are excluded (heavy atoms only). Element column (77:78) used when
    present; otherwise inferred from the atom name's first non-digit char.
    """
    out: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            elem = line[76:78].strip()
            if not elem:
                # Fallback: strip leading digits from atom_name
                a = atom_name.lstrip("0123456789")
                elem = a[0] if a else ""
            if elem.upper() == "H":
                continue
            ch = line[21]
            resnum = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            out.setdefault(ch, []).append((resnum, atom_name, xyz))
    return out


def compute_interface_coverage(
    pred_pdb,
    hotspots_sample,
    binder_chain,
    distance_threshold=GENIE3_INTERFACE_DIST_A,
):
    """Fraction of hotspot residues with any heavy atom within ``distance_threshold``
    of any binder heavy atom in the predicted complex.

    ``hotspots_sample``: list of ``(chain, resnum)`` using the predicted PDB's
    chain/residue numbering.
    """
    atoms = _atoms_per_chain(pred_pdb)
    binder_xyz = np.array([a[2] for a in atoms.get(binder_chain, [])])
    if len(binder_xyz) == 0 or not hotspots_sample:
        return 0.0, 0, 0
    n_total = len(hotspots_sample)
    n_within = 0
    for ch, resnum in hotspots_sample:
        residue_atoms = [a for a in atoms.get(ch, []) if a[0] == resnum]
        if not residue_atoms:
            continue
        rxyz = np.array([a[2] for a in residue_atoms])
        # min distance from any residue atom to any binder atom
        d = np.linalg.norm(rxyz[:, None, :] - binder_xyz[None, :, :], axis=-1)
        if d.min() <= distance_threshold:
            n_within += 1
    return n_within / n_total, n_within, n_total


def interface_pass(frac, n_total):
    if n_total <= 3:
        return frac >= 1.0 - 1e-6
    return frac >= GENIE3_INTERFACE_FRAC - 1e-6


# -----------------------------------------------------------------------------
# Complex RMSD: predicted vs designed, target-aligned, CA-only
# -----------------------------------------------------------------------------


def _ca_coords_by_chain(pdb_path):
    """Return ``dict chain → np.ndarray[N_res, 3]`` of CA coordinates in residue
    order, plus a parallel ``dict chain → list[resnum]``."""
    coords: dict[str, list[np.ndarray]] = {}
    resnums: dict[str, list[int]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch = line[21]
            resnum = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            coords.setdefault(ch, []).append(xyz)
            resnums.setdefault(ch, []).append(resnum)
    return {c: np.stack(v) for c, v in coords.items()}, resnums


def _kabsch(A, B):
    """Return rotation matrix R and translation t such that R @ A.T + t ≈ B.T."""
    Ac = A.mean(0)
    Bc = B.mean(0)
    A0 = A - Ac
    B0 = B - Bc
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Bc - R @ Ac
    return R, t


def _kabsch_transform(pred_target_xyz, des_target_xyz):
    """Return (R, t) such that ``(R @ pred.T).T + t`` approximates ``des``.

    Uses the smaller of the two CA arrays (truncates to the shorter length).
    """
    n = min(len(pred_target_xyz), len(des_target_xyz))
    R, t = _kabsch(pred_target_xyz[:n], des_target_xyz[:n])
    return R, t, n


def _concat_target_ca(coords_by_chain, target_chains):
    """Concatenate CA coords for the given chain list (in order)."""
    return np.concatenate([coords_by_chain[c] for c in target_chains if c in coords_by_chain])


def complex_rmsd_target_aligned(designed_pdb, predicted_pdb, target_chains_sample, binder_chain):
    """Align ``predicted_pdb`` onto ``designed_pdb`` over target CAs, then return
    CA-RMSD of the binder.

    Colabdesign's binder protocol writes the predicted complex with the FULL
    target concatenated into a single chain ``A`` (regardless of input chain
    count) and the binder in chain ``B``. We concatenate the designed target
    chains in ``target_chains_sample`` order to match.
    """
    des_coords, _ = _ca_coords_by_chain(designed_pdb)
    pred_coords, _ = _ca_coords_by_chain(predicted_pdb)

    pred_chain_ids = list(pred_coords.keys())
    if len(pred_chain_ids) < 2:
        return float("nan")
    pred_target_chain = pred_chain_ids[0]
    pred_binder_chain = pred_chain_ids[1]

    target_des = _concat_target_ca(des_coords, target_chains_sample)
    target_pred = pred_coords[pred_target_chain]

    R, t, _ = _kabsch_transform(target_pred, target_des)
    pred_binder_xyz = pred_coords[pred_binder_chain]
    des_binder_xyz = des_coords[binder_chain]
    n = min(len(pred_binder_xyz), len(des_binder_xyz))
    pred_binder_aligned = (R @ pred_binder_xyz[:n].T).T + t
    diff = pred_binder_aligned - des_binder_xyz[:n]
    rmsd = float(np.sqrt((diff ** 2).sum(-1).mean()))
    return rmsd


# -----------------------------------------------------------------------------
# AF2 prediction
# -----------------------------------------------------------------------------


def _make_af_model(use_initial_guess):
    """Create a fresh AF2-Multimer binder model."""
    from colabdesign import mk_afdesign_model

    return mk_afdesign_model(
        protocol="binder",
        use_multimer=True,
        use_initial_guess=use_initial_guess,
        use_initial_atom_pos=False,
        data_dir=AF2_PARAM_DIR,
    )


def _extract_target_pdb(target_path, target_orig_chains, out_path):
    """Write a clean target-only PDB containing exactly ``target_orig_chains``.

    Some downloaded target PDBs may contain HETATM or other chains; we keep
    ATOM records for the requested chains only.
    """
    keep = set(target_orig_chains)
    with open(target_path) as f, open(out_path, "w") as g:
        for line in f:
            if line.startswith("ATOM") and line[21] in keep:
                g.write(line)
            elif line.startswith("TER"):
                g.write(line)
        g.write("END\n")


def predict_complex_af2(
    af_model,
    target_pdb,
    target_chains_str,
    binder_len,
    binder_seq,
    models,
    num_recycles,
):
    """Run AF2-Multimer prediction for a (target, binder_seq) pair.

    Returns ``dict`` with ``plddt``, ``ptm``, ``i_ptm``, ``pae_mean``, ``i_pae_mean``,
    ``pae_min``, ``i_pae_min`` (Å), ``binder_len``, ``target_len``, ``aux``.
    """
    af_model.prep_inputs(
        pdb_filename=target_pdb,
        chain=target_chains_str,
        binder_len=binder_len,
        use_binder_template=False,
        rm_target_seq=False,
        rm_target_sc=False,
    )
    result = None
    best = None
    for m in models:
        af_model.predict(seq=binder_seq, models=[m], num_recycles=num_recycles, verbose=False)
        aux = af_model.aux
        log = aux["log"]
        pae_full = np.asarray(aux["pae"])  # NxN in Å
        target_len = pae_full.shape[0] - binder_len
        # interaction PAE: binder rows × target cols (and target rows × binder cols)
        ipae_block_a = pae_full[target_len:, :target_len]
        ipae_block_b = pae_full[:target_len, target_len:]
        ipae_combined = np.concatenate([ipae_block_a.flatten(), ipae_block_b.flatten()])
        metrics = {
            "model_num": int(m),
            "plddt": float(log.get("plddt", float("nan"))),
            "ptm": float(log.get("ptm", float("nan"))),
            "i_ptm": float(log.get("i_ptm", float("nan"))),
            "pae_mean_norm": float(log.get("pae", float("nan"))),
            "i_pae_mean_norm": float(log.get("i_pae", float("nan"))),
            "pae_mean": float(pae_full.mean()),
            "i_pae_mean": float(ipae_combined.mean()),
            "i_pae_min": float(ipae_combined.min()),
            "target_len": int(target_len),
            "binder_len": int(binder_len),
        }
        if best is None or metrics["i_ptm"] > best["i_ptm"]:
            best = metrics
            result = (m, aux)
    best_model, best_aux = result
    return best, best_model, best_aux


# -----------------------------------------------------------------------------
# Per-sample workflow
# -----------------------------------------------------------------------------


def find_target_pdb(target_dir, motif_name, target_orig_chains):
    """Return the genie3 target PDB for ``motif_name``.

    The per-chain files (e.g. ``01_bhrf1-chain_B.pdb``) in the genie3 dataset are
    empty placeholders, so we always use the full ``<motif>.pdb`` and filter
    chains downstream.
    """
    full = target_dir / f"{motif_name}.pdb"
    if full.exists():
        return full
    raise FileNotFoundError(f"No target PDB found for {motif_name} in {target_dir}")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", required=True, type=Path,
                    help="Root dir with per-motif subdirs (e.g. .../genie3_benchmark/cc141-...)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=GENIE3_CONFIG,
                    help="Genie3 benchmark YAML (defaults to examples/sampling/genie3_benchmark.yaml)")
    ap.add_argument("--targets-dir", type=Path, default=GENIE3_TARGET_DIR)
    ap.add_argument("--seq-design", nargs="+", default=["proteinmpnn"],
                    choices=["proteinmpnn", "caliby"],
                    help="Which sequence designers to run (space-separated)")
    ap.add_argument("--num-seqs", type=int, default=1, help="Sequences per sample per designer")
    ap.add_argument("--num-models", type=int, default=1, help="AF2 multimer models to run (1-5)")
    ap.add_argument("--num-recycles", type=int, default=3)
    ap.add_argument("--num-samples-per-motif", type=int, default=None,
                    help="Limit samples per motif (for quick tests)")
    ap.add_argument("--motifs", nargs="+", default=None, help="Restrict to these motif names")
    ap.add_argument("--temperature", type=float, default=0.1, help="ProteinMPNN sampling temperature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caliby-conformers", type=int, default=8, help="Conformers per sample for caliby ensemble mode")
    ap.add_argument("--caliby-no-ensemble", action="store_true",
                    help="Disable ensemble-conditioned caliby (use single-structure sampling)")
    ap.add_argument("--skip-seq-design", action="store_true",
                    help="Use existing seq_design results in output-dir (skip running ProteinMPNN/Caliby)")
    ap.add_argument("--use-initial-guess", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true",
                    help="Skip (motif, sample, designer, seq_idx) rows already present in output-dir/results.csv")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    genie3_cfg = load_genie3_config(args.config)

    # Discover motifs available in samples dir.
    motif_dirs = []
    for d in sorted(args.samples_dir.iterdir()):
        if not d.is_dir():
            continue
        if args.motifs and d.name not in args.motifs:
            continue
        if d.name not in genie3_cfg:
            print(f"[warn] motif {d.name} not in config, skipping")
            continue
        motif_dirs.append(d)

    # Gather all samples + parse chain info.
    sample_records = []  # one row per (motif, sample_pdb)
    for d in motif_dirs:
        motif = d.name
        contig = genie3_cfg[motif]["contig"]
        hotspot_str = genie3_cfg[motif]["hotspots"]
        orig_chains = parse_contig_target_chains(contig)

        sample_pdbs = sorted(d.glob(f"sample_{motif}_*.pdb"))
        if args.num_samples_per_motif:
            sample_pdbs = sample_pdbs[: args.num_samples_per_motif]
        for sp in sample_pdbs:
            target_chains, binder_chain, orig_to_sample, chains_dict = split_sample_chains(sp, orig_chains)
            binder_len = len(chains_dict[binder_chain])
            hotspots_sample = parse_hotspots(hotspot_str, orig_to_sample)
            sample_records.append({
                "motif": motif,
                "sample_pdb": str(sp),
                "key": sp.stem,
                "binder_chain": binder_chain,
                "target_chains_sample": target_chains,
                "binder_len": binder_len,
                "hotspots_sample": hotspots_sample,
                "orig_to_sample": orig_to_sample,
                "chains_dict": {c: chains_dict[c] for c in chains_dict},
                "fixed_pos_seq": build_fixed_pos_seq(target_chains, chains_dict),
            })

    if not sample_records:
        print("[error] no samples found")
        sys.exit(1)

    print(f"Discovered {len(sample_records)} samples across {len(motif_dirs)} motifs")

    # -------------------------------------------------------------------------
    # Sequence design (subprocess to other envs).
    # -------------------------------------------------------------------------
    seq_results = {}  # designer → list[dict(key, binder_seqs)]

    seq_design_dir = args.output_dir / "seq_design"
    seq_design_dir.mkdir(parents=True, exist_ok=True)
    seq_records_path = seq_design_dir / "seq_records.json"

    if not args.skip_seq_design:
        with open(seq_records_path, "w") as f:
            json.dump(sample_records, f, indent=2, default=str)

    if "proteinmpnn" in args.seq_design:
        out = seq_design_dir / "proteinmpnn"
        cache = out / "_pmpnn_results.json"
        if args.skip_seq_design and cache.exists():
            with open(cache) as f:
                seq_results["proteinmpnn"] = json.load(f)["results"]
        else:
            pmpnn_in = [
                {"pdb": r["sample_pdb"], "key": r["key"], "binder_chain": r["binder_chain"]}
                for r in sample_records
            ]
            print(f"[seq_design] running ProteinMPNN on {len(pmpnn_in)} samples")
            seq_results["proteinmpnn"] = run_proteinmpnn(
                pmpnn_in, args.num_seqs, args.temperature, args.seed, out
            )

    if "caliby" in args.seq_design:
        out = seq_design_dir / "caliby"
        cache = out / "_caliby_results.json"
        if args.skip_seq_design and cache.exists():
            with open(cache) as f:
                seq_results["caliby"] = json.load(f)["results"]
        else:
            caliby_in = [
                {
                    "pdb": r["sample_pdb"],
                    "key": r["key"],
                    "binder_chain": r["binder_chain"],
                    "fixed_pos_seq": r["fixed_pos_seq"],
                }
                for r in sample_records
            ]
            print(f"[seq_design] running Caliby on {len(caliby_in)} samples")
            seq_results["caliby"] = run_caliby(
                caliby_in,
                args.num_seqs,
                args.caliby_conformers,
                use_ensemble=not args.caliby_no_ensemble,
                out_dir=out,
            )

    # Index seq results by key.
    by_key_per_designer = {}
    for designer, results in seq_results.items():
        by_key_per_designer[designer] = {r["key"]: r["binder_seqs"] for r in results}

    # -------------------------------------------------------------------------
    # AF2 prediction.
    # -------------------------------------------------------------------------
    print("[af2] importing colabdesign / jax")
    from colabdesign import mk_afdesign_model  # noqa: F401

    af2_dir = args.output_dir / "af2_pred"
    af2_dir.mkdir(parents=True, exist_ok=True)

    model_nums = list(range(args.num_models))

    target_pdb_cache = {}  # motif → cleaned target pdb path

    rows = []
    af_model_cache = {}  # (target_chains_str, binder_len) → model instance
    last_prep_key = None

    # Sort by (motif, binder_len) so the JAX-compiled model is reused across
    # consecutive samples of the same shape — recompilation per length is the
    # main runtime cost.
    sample_records.sort(key=lambda r: (r["motif"], r["binder_len"], r["key"]))

    # Resume: load existing results.csv, skip already-done rows.
    results_path = args.output_dir / "results.csv"
    done_keys = set()
    if args.resume and results_path.exists():
        prev = pd.read_csv(results_path)
        rows = prev.to_dict("records")
        for row in rows:
            done_keys.add((row["motif"], row["sample_key"], row["designer"], int(row["seq_idx"])))
        print(f"[resume] loaded {len(rows)} prior rows; skipping done (motif,sample,designer,seq_idx) keys")

    for r in sample_records:
        motif = r["motif"]
        contig = genie3_cfg[motif]["contig"]
        orig_chains = parse_contig_target_chains(contig)
        if motif not in target_pdb_cache:
            tpath = find_target_pdb(args.targets_dir, motif, orig_chains)
            cleaned = af2_dir / f"target_{motif}.pdb"
            _extract_target_pdb(tpath, orig_chains, cleaned)
            target_pdb_cache[motif] = cleaned
        target_pdb = target_pdb_cache[motif]
        target_chains_orig_str = ",".join(orig_chains)

        for designer, binder_seqs_by_key in by_key_per_designer.items():
            binder_seqs = binder_seqs_by_key.get(r["key"], [])
            if not binder_seqs:
                print(f"[warn] no {designer} sequences for {r['key']}")
                continue
            for seq_idx, binder_seq in enumerate(binder_seqs):
                if binder_seq is None or len(binder_seq) != r["binder_len"]:
                    print(f"[warn] {designer}/{r['key']}/{seq_idx}: bad binder seq len "
                          f"{len(binder_seq) if binder_seq else None} vs expected {r['binder_len']}")
                    continue
                if (motif, r["key"], designer, seq_idx) in done_keys:
                    continue
                prep_key = (motif, r["binder_len"])
                if prep_key != last_prep_key:
                    # Re-create model to drop cached state.
                    af_model = _make_af_model(args.use_initial_guess)
                    af_model_cache.clear()
                    af_model_cache[prep_key] = af_model
                    last_prep_key = prep_key
                else:
                    af_model = af_model_cache[prep_key]

                best_metrics, best_model, best_aux = predict_complex_af2(
                    af_model,
                    str(target_pdb),
                    target_chains_orig_str,
                    r["binder_len"],
                    binder_seq,
                    model_nums,
                    args.num_recycles,
                )

                # Save predicted complex PDB.
                pred_pdb = af2_dir / f"{designer}_{r['key']}_seq{seq_idx}_model{best_metrics['model_num']}.pdb"
                af_model.save_pdb(str(pred_pdb))

                # Interface coverage on predicted complex.
                # Predicted PDB merges multi-chain targets into a single chain A
                # with a +50-residue gap between adjacent target chains.
                target_chain_lengths = target_chain_lengths_from_target_pdb(
                    target_pdb, orig_chains
                )
                hotspots_predicted = map_hotspots_to_predicted(
                    genie3_cfg[motif]["hotspots"], target_chain_lengths
                )
                iface_frac, n_within, n_total = compute_interface_coverage(
                    str(pred_pdb),
                    hotspots_predicted,
                    _binder_chain_predicted(pred_pdb, 1),
                )
                iface_ok = interface_pass(iface_frac, n_total) if n_total > 0 else None

                # Complex RMSD: predicted vs designed (target-aligned binder CA).
                rmsd = complex_rmsd_target_aligned(
                    r["sample_pdb"], str(pred_pdb), r["target_chains_sample"], r["binder_chain"]
                )

                # Genie3 thresholds. "Binder pTM" in genie3 = interface pTM.
                ptm_pass = best_metrics["i_ptm"] >= GENIE3_PTM_MIN
                ipae_pass = best_metrics["i_pae_min"] <= GENIE3_IPAE_MAX
                rmsd_pass = rmsd <= GENIE3_RMSD_MAX if not np.isnan(rmsd) else False
                criteria_passes = [ptm_pass, ipae_pass, rmsd_pass]
                if iface_ok is not None:
                    criteria_passes.append(iface_ok)
                overall_pass = all(criteria_passes)

                row = {
                    "motif": motif,
                    "sample_key": r["key"],
                    "sample_pdb": r["sample_pdb"],
                    "designer": designer,
                    "seq_idx": seq_idx,
                    "binder_seq": binder_seq,
                    "binder_len": r["binder_len"],
                    "target_chains_orig": target_chains_orig_str,
                    "target_chains_sample": ",".join(r["target_chains_sample"]),
                    "binder_chain_sample": r["binder_chain"],
                    "af2_model_num": best_metrics["model_num"],
                    "plddt": best_metrics["plddt"],
                    "ptm": best_metrics["ptm"],
                    "i_ptm": best_metrics["i_ptm"],
                    "pae_mean": best_metrics["pae_mean"],
                    "i_pae_mean": best_metrics["i_pae_mean"],
                    "i_pae_min": best_metrics["i_pae_min"],
                    "complex_rmsd": rmsd,
                    "iface_frac": iface_frac,
                    "iface_within": n_within,
                    "iface_total": n_total,
                    "pred_pdb": str(pred_pdb),
                    "genie3_ptm_pass": ptm_pass,
                    "genie3_ipae_pass": ipae_pass,
                    "genie3_rmsd_pass": rmsd_pass,
                    "genie3_iface_pass": iface_ok,
                    "genie3_pass": overall_pass,
                }
                rows.append(row)
                print(
                    f"[af2] {motif}/{designer}/{r['key']}/seq{seq_idx}: "
                    f"i_ptm={best_metrics['i_ptm']:.3f} i_pae_min={best_metrics['i_pae_min']:.2f} "
                    f"rmsd={rmsd:.2f} iface={iface_frac:.2f} ({n_within}/{n_total}) pass={overall_pass}"
                )

                # Persist intermediate CSV after each row.
                pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)

    pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)
    print(f"Wrote {len(rows)} rows to {args.output_dir / 'results.csv'}")


def _binder_chain_predicted(pred_pdb, n_target_chains):
    """Determine binder chain in the predicted complex (chain after the targets)."""
    chains = chain_residue_counts(pred_pdb)
    chain_ids = list(chains.keys())
    if len(chain_ids) <= n_target_chains:
        return chain_ids[-1]
    return chain_ids[n_target_chains]


if __name__ == "__main__":
    main()
