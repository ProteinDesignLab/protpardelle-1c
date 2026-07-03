"""Build dn9 sampling configs for cc148 (unindexed) and cc149 (indexed).

Diversity axes built into each config (sweep is product of motif rows × search_space):

  - Axis A (Architector geometry): 5 main + 3 scatter geometries via motif PDB
  - Axis B (Coordination strategy): bb-O variants per geometry (base/bbo1/bbo2/drop1)
  - Axis C (Solvent exposure): partial proxy via `binder_sse_prop` (helical = more
    buried/amphipathic shell, beta = more exposed/edge) and `motif_sse` (motif-local
    SSE: H/E)
  - SSE flank: cc148/cc149 motif_sse_flank=3 to enforce motif-adjacent SSE

Each row is (motif_variant, motif_sse, binder_sse_prop, total_lengths).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml


# 5 main + 3 scatter geometries. Tuples: (geom_key, REE shortlist).
# For each, we use the LANTHANIDE-specific PDB. Multiple REEs give extra
# rotational/translational diversity at the motif level.
MAIN_GEOMS = [
    ("am_c3_7H2O_c0", ["la", "nd"]),                              # CN7
    ("pentagonal_bipyramidal", ["la", "nd"]),                     # CN7
    ("am_c3_9H2O_c0", ["la", "nd"]),                              # CN9
    ("tricapped_trigonal_prismatic", ["la", "nd"]),               # CN9
    ("axial_bicapped_hexagonal_planar", ["la"]),                  # CN10
]
SCATTER_GEOMS = [
    ("capped_octahedral", ["la", "nd"]),                          # CN7
    ("YICLED", ["la", "dy"]),                                     # CN9 scatter
]

# Allocation per dn9.md axis A (used by select-final stage; here it informs
# how aggressively to oversample each).
GEOM_BUDGET = {
    "am_c3_7H2O_c0": 35,
    "pentagonal_bipyramidal": 30,
    "am_c3_9H2O_c0": 30,
    "tricapped_trigonal_prismatic": 25,
    "axial_bicapped_hexagonal_planar": 35,
    "capped_octahedral": 12,
    "YICLED": 12,
}

# Per-geom bb-O variant pool. base = original (acetate-tip + HOH-bbO).
# bbo1 = 1 ASP→HOH (one extra bb-O anchor).
# drop1 = drop the most-outer HOH (let solvent fill it).
VARIANTS = ["base", "bbo1", "drop1"]

# Per-geom motif-local SSE. H/E gives helix-flanked vs strand-flanked binding sites.
MOTIF_SSE_OPTIONS = ["H", "E"]

# Global binder SSE proportions [helix, sheet, loop]. dn9.md says SS-flanking is
# correlated with exposure, so we keep two anchors: alpha-rich (compact, buried)
# vs beta-rich (more exposed edges).
BINDER_SSE_OPTIONS = [
    ("a", [0.7, 0.0, 0.3]),  # alpha-rich (more buried, helical)
    ("b", [0.1, 0.5, 0.4]),  # beta-rich (more exposed, sheet)
]

# Length range stand-in for solvent exposure: short (more exposed surface area
# per binding-site residue) vs long (more buried).
LENGTH_RANGES = [
    ("m", [[100, 160]]),
]


def find_variant_pdb(motif_dir: Path, ree: str, geom: str, variant: str) -> Path | None:
    """Resolve the variant PDB file for a given REE × geometry × variant."""
    # Filenames embed coordination context; lookup heuristically.
    candidates = list(motif_dir.glob(f"{ree}_acetate*{geom}*__{variant}.pdb"))
    if not candidates:
        # Try the cn10 acetate4_water2 / acetate5 special name
        candidates = list(motif_dir.glob(f"{ree}_acetate*{geom}*__{variant}.pdb"))
    if not candidates:
        return None
    # Prefer the acetate+water family (more biologically natural with HOH).
    candidates.sort(key=lambda p: ("acetate_water" in p.name, "acetate5" in p.name), reverse=True)
    return candidates[0]


def build_rows(motif_dir: Path):
    rows = []  # (name, motif_basename_no_pdb, motif_sse, binder_sse_prop, total_lengths, geom_key, cell_key)
    all_geoms = MAIN_GEOMS + SCATTER_GEOMS
    for geom_key, ree_list in all_geoms:
        for ree, variant in itertools.product(ree_list, VARIANTS):
            pdb = find_variant_pdb(motif_dir, ree, geom_key, variant)
            if pdb is None:
                continue
            motif_basename = pdb.stem
            # Iterate motif SSE and global SSE proportions and length ranges
            sse_pool = MOTIF_SSE_OPTIONS
            bsse_pool = BINDER_SSE_OPTIONS
            len_pool = LENGTH_RANGES
            # For scatter geoms, take a subset to stay within ~12 final shots
            if geom_key in [g for g, _ in SCATTER_GEOMS]:
                sse_pool = ["H"]
                bsse_pool = BINDER_SSE_OPTIONS[:1]
                len_pool = LENGTH_RANGES[:1]
            for msse in sse_pool:
                for blbl, bprop in bsse_pool:
                    for llbl, lrng in len_pool:
                        cell_key = f"{geom_key}__{variant}__sse-{msse}__bs-{blbl}__len-{llbl}"
                        rows.append({
                            "name": f"{motif_basename}|{msse}|{blbl}|{llbl}",
                            "motif_basename": motif_basename,
                            "motif_sse": msse,
                            "binder_sse_prop": bprop,
                            "total_lengths": lrng,
                            "geom_key": geom_key,
                            "cell_key": cell_key,
                        })
    return rows


def write_yaml(rows, out_path: Path, model_name: str, model_epoch: int, sampling_subcfg: str):
    n = len(rows)
    cfg = {
        "search_space": {
            "models": [[model_name, str(model_epoch), sampling_subcfg]],
            "step_scales": [1.2],
            "schurns": [200],
            "crop_cond_starts": [0.0],
            "translations": [[0.0, 0.0, 0.0]],
        },
        "motifs": [r["motif_basename"] for r in rows],
        "motif_contigs": ["unindexed" if model_name == "cc148" else None] * n,
        "total_lengths": [r["total_lengths"] for r in rows],
        "hotspots": [None] * n,
        "ssadj": [None] * n,
        "motif_sse": [r["motif_sse"] for r in rows],
        "binder_sse_prop": [r["binder_sse_prop"] for r in rows],
        "motif_sse_flank": 3,
    }

    if model_name == "cc149":
        # For cc149 (indexed), we need motif_contigs. Build a generic one:
        # "5-30;A2-2;3-9;A3-3;3-9;...;A{K-1}-{K-1};3-15" — discontiguous placement.
        # For each motif we need to know how many residues it has post-HOH-injection.
        # The architector base PDBs have N ASPs + M HOHs. Variants change this:
        #   __base : N ASPs (HOH injection adds M residues via inject_hoh_as_bb_o)
        #   __bbo1 : N-1 ASPs (+1 was converted to HOH source) — actual HOH count rises by 1
        #   __bbo2 : N-2 ASPs (+2 ASPs converted to HOH)
        #   __drop1: N ASPs, M-1 HOHs
        # We don't know exact counts without parsing each PDB. Use a heuristic
        # contig with K=8 motif residues max (covers 4 ASP + 4 HOH).
        # The sample.py loader will downcount actual K from feats and the placement
        # will adjust gracefully.
        motif_contigs = []
        for r in rows:
            # Conservative: build contig with up to 9 motif slots; we err on
            # the side of "too many" and sample.py will use however many feats
            # have. Actually for cc149 (concat+condmask) the contig MUST match
            # exactly. Use unindexed-style placement is not allowed; we use the
            # "fixed-spacing" pattern with min/max gaps that have worked before.
            # Need to count actual motif residues from the source PDB.
            pdb_path = (Path("/home/users/tianyulu/code/protpardelle-1c/examples/motifs/dn9_variants")
                        / f"{r['motif_basename']}.pdb")
            n_motif_res = _count_motif_residues(pdb_path)
            # Build wider scaffold gaps so total length can reach 100-160.
            # Pattern: "30-60;A2-2;5-15;A3-3;...;A{K+1}-{K+1};30-60"
            parts = [f"30-60"]
            for k in range(n_motif_res):
                parts.append(f"A{k+2}-{k+2}")
                if k < n_motif_res - 1:
                    parts.append("5-15")
            parts.append("30-60")
            motif_contigs.append(";".join(parts))
        cfg["motif_contigs"] = motif_contigs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    print(f"Wrote {out_path} with {n} motif rows")


def _count_motif_residues(pdb_path: Path) -> int:
    """Count number of motif residues after HOH-injection (ASPs + HOHs)."""
    if not pdb_path.exists():
        return 4  # fallback
    residues = set()
    for ln in pdb_path.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            resname = ln[17:20].strip()
            if resname in ("LA", "ND", "DY", "LU", "PR", "SM", "EU", "TB", "HO", "ER", "TM", "YB", "CE", "GD", "PM"):
                continue
            chain = ln[21]
            resseq = ln[22:26].strip()
            residues.add((chain, resseq, resname))
    return len(residues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = build_rows(args.motif_dir)
    print(f"Total rows: {len(rows)}")
    # Print summary per cell
    from collections import Counter
    cell_counts = Counter([r["cell_key"] for r in rows])
    print(f"Total unique cells: {len(cell_counts)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # cc148 (unindexed)
    write_yaml(
        rows,
        args.out_dir / "dn9_cc148.yaml",
        model_name="cc148",
        model_epoch=202,
        sampling_subcfg="sampling_acidic_tip_water_bbO_condmask_allatom_s1_caliby",
    )

    # cc149 (concat+condmask)
    write_yaml(
        rows,
        args.out_dir / "dn9_cc149.yaml",
        model_name="cc149",
        model_epoch=172,
        sampling_subcfg="sampling_acidic_tip_water_bbO_condmask_allatom_s1_caliby",
    )

    # Save the row → cell map for use during filtering/selection
    import json
    row_map = {r["name"]: r for r in rows}
    with (args.out_dir / "dn9_row_map.json").open("w") as f:
        json.dump(row_map, f, indent=2)
    print(f"Wrote row map to {args.out_dir/'dn9_row_map.json'}")


if __name__ == "__main__":
    main()
