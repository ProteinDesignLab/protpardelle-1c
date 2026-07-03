"""Final cell-balanced selection of dn9 designs.

Combines:
  - Filter CSV (with pI, K+E, RoG, XXX, passed flag)
  - RF3 predictions CSV (mean pLDDT)
  - dn9 row map (cell_key per sample, from config build step)

Applies dn9.md axis-A budgets (am_c3_7H2O=35, pentagonal_bip=30, am_c3_9H2O=30,
tricapped_trig=25, axial_bicapped_hex=35, scatter=35). Within each axis-A cell,
distribute across coordination strategy (axis B) and exposure proxy (axis C)
cells. Within each final cell, rank by:
  1. Mean pLDDT > 50 (hard filter, post-hoc — does NOT rank)
  2. Tip-atom RMSD to motif (lower better) — not yet computed; placeholder
  3. Backbone outliers
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


GEOM_BUDGET = {
    "am_c3_7H2O_c0": 35,
    "pentagonal_bipyramidal": 30,
    "am_c3_9H2O_c0": 30,
    "tricapped_trigonal_prismatic": 25,
    "axial_bicapped_hexagonal_planar": 35,
    "capped_octahedral": 12,
    "YICLED": 12,
}
TOTAL_BUDGET = 190


def derive_cell_key_from_pdb_name(pdb_name: str, row_map: dict) -> tuple[str, str]:
    """From a PDB filename, extract motif_basename and look up cell_key.

    Sample PDB names follow pattern:
      sample_<motif_basename>__sse-<sse>_<idx>.pdb
    or:
      <motif_basename>__sse-<sse>/sample_*.pdb

    We return (geom_key, full_cell_key) by matching to row_map.
    """
    # Strip .pdb
    stem = pdb_name.rsplit(".", 1)[0]
    # Try to match against any row in row_map
    # The motif_basename is what we wrote in the config; the per-sample save path
    # contains it. Iterate row_map keys and look for substring match.
    for name, row in row_map.items():
        motif_basename = row["motif_basename"]
        if motif_basename in stem:
            return row["geom_key"], row["cell_key"]
    return "unknown", "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-csv", type=Path, required=True)
    ap.add_argument("--rf3-csv", type=Path, default=None)
    ap.add_argument("--row-map", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--plddt-min", type=float, default=50.0)
    args = ap.parse_args()

    df = pd.read_csv(args.filter_csv)
    df = df[df["passed"] == True].copy()
    print(f"After filter pass: {len(df)} candidates")

    if args.rf3_csv is not None and args.rf3_csv.exists():
        rf3 = pd.read_csv(args.rf3_csv)
        df = df.merge(rf3[["pdb", "mean_plddt"]], on="pdb", how="left")
        # post-hoc pLDDT filter
        df = df[df["mean_plddt"].fillna(0) > args.plddt_min].copy()
        print(f"After pLDDT > {args.plddt_min}: {len(df)} candidates")
    else:
        df["mean_plddt"] = np.nan
        print("[WARN] no RF3 CSV provided — skipping pLDDT filter")

    with args.row_map.open() as f:
        row_map = json.load(f)

    # Attach cell info
    df["geom_key"], df["cell_key"] = zip(*df["pdb"].apply(lambda x: derive_cell_key_from_pdb_name(x, row_map)))

    print(f"\nPer-geom counts (passed pLDDT):")
    print(df.groupby("geom_key").size().to_string())

    # Allocate budget per geom, picking best samples by:
    # - lower rog_norm (more globular)
    # - pI distance from 4.5 (target acidic but not too extreme)
    # - has_xxx_triple (penalize repetitive caliby sequences but not disqualify)
    # - mean_plddt (if available — higher is better)
    # - frac_G (penalize polyglycine sequences — strong signal of bad MPNN output)
    df["frac_G"] = df["sequence"].apply(
        lambda s: (s.count("G") / max(1, len(s))) if isinstance(s, str) else 1.0
    )
    df["seq_unique_aa"] = df["sequence"].apply(
        lambda s: len(set(s)) / 20.0 if isinstance(s, str) else 0.0
    )
    df["rank_score"] = (
        df["rog_norm"]
        + 0.5 * abs(df["pi"] - 4.5)
        + 0.5 * df["has_xxx_triple"].astype(float)
        - 0.01 * df["mean_plddt"].fillna(50)
        + 4.0 * df["frac_G"]                 # strong polyG penalty
        - 2.0 * df["seq_unique_aa"]          # reward sequence diversity
    )

    selected = []
    for geom_key, geom_df in df.groupby("geom_key"):
        budget = GEOM_BUDGET.get(geom_key, 12)  # default 12 for unknown/scatter
        # Within geom: round-robin across (variant × SSE) cells, sorted by rank_score
        per_cell = []
        for cell_key, cell_df in geom_df.groupby("cell_key"):
            sorted_cell = cell_df.sort_values("rank_score").reset_index(drop=True)
            per_cell.append(sorted_cell.to_dict(orient="records"))

        picks = []
        # Round-robin pick
        active_cells = [c for c in per_cell if c]
        while active_cells and len(picks) < budget:
            new_active = []
            for cell in active_cells:
                if len(picks) >= budget:
                    new_active.append(cell)
                    continue
                if cell:
                    picks.append(cell.pop(0))
                if cell:
                    new_active.append(cell)
            active_cells = new_active

        for row in picks:
            r = dict(row)
            r["selected_geom_budget"] = budget
            selected.append(r)
        print(f"{geom_key}: budget={budget}, picked={len(picks)}")

    # Add scatter geoms (small fixed allocation)
    scatter_budget = 35 - sum(GEOM_BUDGET.get(g, 0) for g in df["geom_key"].unique()
                              if g not in {"am_c3_7H2O_c0", "pentagonal_bipyramidal", "am_c3_9H2O_c0",
                                           "tricapped_trigonal_prismatic", "axial_bicapped_hexagonal_planar"})

    final_df = pd.DataFrame(selected)
    print(f"\n=== Final selection: {len(final_df)} / {TOTAL_BUDGET} ===")
    if len(final_df) < TOTAL_BUDGET:
        # Top up with best remaining samples from any cell
        rest = df[~df.index.isin(final_df.index)].sort_values("rank_score")
        top_up = rest.head(TOTAL_BUDGET - len(final_df))
        final_df = pd.concat([final_df, top_up], ignore_index=True)
        print(f"Topped up to {len(final_df)} with cross-cell best.")

    final_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    print(f"\nGeom breakdown:")
    print(final_df.groupby("geom_key").size().to_string())


if __name__ == "__main__":
    main()
