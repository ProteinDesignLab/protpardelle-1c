"""Cluster passing genie3 designs by TM-score and plot per-target unique successes.

Inputs the ``results.csv`` produced by ``af2.py``. For each motif:
1. Filter rows with ``genie3_pass == True``.
2. Extract the predicted binder chain from each ``pred_pdb`` into a temp PDB.
3. Pairwise TMalign on binder-only PDBs (TM-score normalized by the shorter chain).
4. Greedy cluster with TM >= 0.6 → same cluster.
5. Plot unique-success counts per target.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TMALIGN_BIN = "/home/groups/possu/software/TMalign"
TM_THRESHOLD = 0.6

_TM_RE = re.compile(r"TM-score=\s*([\d.]+)\s+\(if normalized by length of Chain_[12]")


def extract_binder_pdb(pred_pdb: Path, out_pdb: Path):
    """Write only the binder chain (last chain in the predicted complex) to ``out_pdb``."""
    chains = []
    with open(pred_pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                if ch not in chains:
                    chains.append(ch)
    binder = chains[-1]
    with open(pred_pdb) as f, open(out_pdb, "w") as g:
        for line in f:
            if line.startswith("ATOM") and line[21] == binder:
                g.write(line)
            elif line.startswith("TER"):
                g.write(line)
        g.write("END\n")


def tmalign(pdb_a: Path, pdb_b: Path) -> float:
    """Return the smaller of the two normalized TM-scores from TMalign output."""
    out = subprocess.run([TMALIGN_BIN, str(pdb_a), str(pdb_b)], capture_output=True, text=True, check=True).stdout
    scores = [float(m.group(1)) for m in _TM_RE.finditer(out)]
    if not scores:
        return float("nan")
    return min(scores)


def greedy_cluster(pdb_paths: list[Path], threshold: float = TM_THRESHOLD) -> list[int]:
    """Return cluster labels (0-indexed) using greedy clustering on TM-scores."""
    labels = [-1] * len(pdb_paths)
    cluster_reps: list[int] = []
    for i, p in enumerate(pdb_paths):
        joined = False
        for rep_idx in cluster_reps:
            tm = tmalign(p, pdb_paths[rep_idx])
            if tm >= threshold:
                labels[i] = labels[rep_idx]
                joined = True
                break
        if not joined:
            labels[i] = len(cluster_reps)
            cluster_reps.append(i)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tm-threshold", type=float, default=TM_THRESHOLD)
    ap.add_argument("--designer", default=None, help="Filter to a single designer (e.g. caliby)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results_csv)
    if args.designer:
        df = df[df["designer"] == args.designer]

    motif_order = sorted(df["motif"].unique())
    summary_rows = []
    cluster_rows = []
    bin_dir = args.out_dir / "binder_pdbs"
    bin_dir.mkdir(exist_ok=True)

    for motif in motif_order:
        sub = df[(df["motif"] == motif) & (df["genie3_pass"] == True)].reset_index(drop=True)  # noqa: E712
        if sub.empty:
            summary_rows.append({"motif": motif, "n_passing": 0, "n_unique": 0})
            continue
        pdb_paths = []
        for i, row in sub.iterrows():
            stub = bin_dir / f"{motif}_{i:03d}.pdb"
            extract_binder_pdb(Path(row["pred_pdb"]), stub)
            pdb_paths.append(stub)
        labels = greedy_cluster(pdb_paths, threshold=args.tm_threshold)
        n_unique = len(set(labels))
        for i, lbl in enumerate(labels):
            cluster_rows.append({
                "motif": motif,
                "passing_idx": i,
                "cluster_id": lbl,
                "pred_pdb": str(sub.iloc[i]["pred_pdb"]),
                "sample_key": sub.iloc[i]["sample_key"],
                "designer": sub.iloc[i]["designer"],
                "binder_pdb": str(pdb_paths[i]),
            })
        summary_rows.append({"motif": motif, "n_passing": len(sub), "n_unique": n_unique})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(args.out_dir / "clusters.csv", index=False)

    # Bar plot
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(motif_order)), 4))
    x = np.arange(len(motif_order))
    width = 0.4
    ax.bar(x - width / 2, summary["n_passing"], width, label="Passing", color="#6baed6")
    ax.bar(x + width / 2, summary["n_unique"], width, label=f"Unique (TM<{args.tm_threshold})", color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["motif"], rotation=30, ha="right")
    ax.set_ylabel("Designs")
    title = "Genie3 unique successes per target"
    if args.designer:
        title += f" ({args.designer})"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "unique_successes.png", dpi=150)
    fig.savefig(args.out_dir / "unique_successes.pdf")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
