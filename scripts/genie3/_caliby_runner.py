"""Caliby ensemble-conditioned sequence design runner.

Invoked as a subprocess by ``af2.py`` so it runs in the caliby uv env.

Reads a JSON job spec from --jobs, writes a JSON result to --out.

Job spec schema::

    {
      "samples": [
        {
          "pdb": "/path/to/sample.pdb",
          "key": "sample_01_bhrf1_00",
          "fixed_pos_seq": "A1-157",   # comma-separated ranges in caliby syntax
          "binder_chain": "B"
        },
        ...
      ],
      "num_seqs": 4,
      "num_conformers": 8,
      "use_ensemble": true,
      "ensemble_out_dir": "/path/to/ensembles",
      "design_out_dir": "/path/to/seq_designs"
    }

Result schema::

    {
      "results": [
        {"key": "sample_01_bhrf1_00",
         "full_seqs": [["MAEK...", "GSPK..."], ...],   # list per chain, in order
         "binder_seqs": ["GSPK...", "VVRY...", ...],
         "out_cif": "/path/to/cif"},
        ...
      ]
    }
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def _chain_letters_from_pdb(pdb_path):
    """Return chains in order of first appearance (ATOM records only)."""
    seen = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                if ch not in seen:
                    seen.append(ch)
    return seen


def _chain_letters_from_cif(cif_path):
    """Return chain IDs in order of first appearance for ATOM records in a CIF file.

    Looks for the ``label_asym_id`` field, which in caliby outputs is the 6th
    whitespace-separated column on ATOM lines.
    """
    seen = []
    with open(cif_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            ch = parts[6]
            if ch not in seen:
                seen.append(ch)
    return seen


def _split_caliby_seq(designed_seq, cif_path):
    """Split a Caliby ``seq`` string (chains joined by ``:``) into ``(chain, seq)`` pairs.

    Caliby outputs chains in label_asym_id order, joined by ``:`` literal.
    """
    pieces = designed_seq.split(":")
    chain_order = _chain_letters_from_cif(cif_path)
    if len(pieces) != len(chain_order):
        # Mismatch — fall back to positional assignment with A, B, ...
        chain_order = [chr(ord("A") + i) for i in range(len(pieces))]
    return list(zip(chain_order, pieces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="Job spec JSON")
    ap.add_argument("--out", required=True, help="Result JSON")
    args = ap.parse_args()

    with open(args.jobs) as f:
        job = json.load(f)

    samples = job["samples"]
    num_seqs = int(job.get("num_seqs", 4))
    num_conformers = int(job.get("num_conformers", 8))
    use_ensemble = bool(job.get("use_ensemble", True))
    design_out_dir = Path(job["design_out_dir"])
    design_out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_out_dir = Path(job.get("ensemble_out_dir", design_out_dir / "ensembles"))
    ensemble_out_dir.mkdir(parents=True, exist_ok=True)

    from caliby import (
        clean_pdbs,
        generate_ensembles,
        load_model,
        make_constraints,
        make_ensemble_constraints,
    )

    model = load_model("caliby")

    # Clean inputs (caliby docs strongly recommend this).
    raw_pdbs = [s["pdb"] for s in samples]
    cleaned_dir = design_out_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_pdbs = clean_pdbs(raw_pdbs, out_dir=str(cleaned_dir))

    # Build sample → cleaned path mapping. clean_pdbs returns list aligned by input order.
    sample_to_cleaned = dict(zip(raw_pdbs, cleaned_pdbs))

    # Build constraints dict keyed by pdb_key (caliby derives pdb_key from filename stem).
    constraints_dict = {}
    for s in samples:
        cleaned = sample_to_cleaned[s["pdb"]]
        pdb_key = Path(cleaned).stem
        s["_pdb_key"] = pdb_key
        s["_cleaned_pdb"] = cleaned
        if s.get("fixed_pos_seq"):
            constraints_dict[pdb_key] = {"fixed_pos_seq": s["fixed_pos_seq"]}

    results = []

    if use_ensemble:
        # Generate ensembles for all cleaned pdbs at once.
        pdb_to_conformers = generate_ensembles(
            cleaned_pdbs,
            out_dir=str(ensemble_out_dir),
            num_samples_per_pdb=num_conformers,
            batch_size=min(num_conformers, 8),
        )
        # ``ensemble_sample`` reports ``example_id`` as the stem of the first conformer
        # in each list; map that back to our sample key for downstream lookup.
        primary_stem_to_pdb_key = {
            Path(paths[0]).stem: pdb_key for pdb_key, paths in pdb_to_conformers.items()
        }
        if constraints_dict:
            constraint_df = make_ensemble_constraints(constraints_dict, pdb_to_conformers)
        else:
            constraint_df = None

        ds = model.ensemble_sample(
            pdb_to_conformers,
            num_seqs_per_pdb=num_seqs,
            out_dir=str(design_out_dir / "seqs"),
            pos_constraint_df=constraint_df,
        )
        # Normalize example_ids to our sample keys.
        ds["example_id"] = [
            primary_stem_to_pdb_key.get(eid, eid) for eid in ds["example_id"]
        ]
    else:
        if constraints_dict:
            constraint_df = make_constraints(constraints_dict)
        else:
            constraint_df = None
        ds = model.sample(
            cleaned_pdbs,
            num_seqs_per_pdb=num_seqs,
            out_dir=str(design_out_dir / "seqs"),
            pos_constraint_df=constraint_df,
        )

    # Caliby returns dict with parallel lists: example_id, out_pdb, seq, U, input_seq
    by_key = {}
    for i, ex_id in enumerate(ds["example_id"]):
        by_key.setdefault(ex_id, []).append(i)

    for s in samples:
        pdb_key = s["_pdb_key"]
        binder_chain_sample = s["binder_chain"]
        idxs = by_key.get(pdb_key, [])
        full_seqs_per_chain = []
        binder_seqs = []
        out_cifs = []
        for j in idxs:
            full_seq = ds["seq"][j]
            out_cif = ds["out_pdb"][j]
            chain_pieces = _split_caliby_seq(full_seq, out_cif)
            full_seqs_per_chain.append(chain_pieces)
            binder_seq = None
            for ch, p in chain_pieces:
                if ch == binder_chain_sample:
                    binder_seq = p
                    break
            if binder_seq is None and chain_pieces:
                binder_seq = chain_pieces[-1][1]
            binder_seqs.append(binder_seq)
            out_cifs.append(out_cif)
        results.append(
            {
                "key": s["key"],
                "pdb": s["pdb"],
                "cleaned_pdb": s["_cleaned_pdb"],
                "binder_seqs": binder_seqs,
                "out_cifs": out_cifs,
            }
        )

    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
