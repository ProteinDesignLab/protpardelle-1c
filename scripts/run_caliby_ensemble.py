"""Run Caliby ensemble-conditioned sequence design as a subprocess helper.

Intended to be invoked in Caliby's venv (which has its own protpardelle pin).
Reads args from a JSON file path (one arg) to keep the CLI simple.

Two input modes are supported:

(A) Legacy single-job mode:
{
    "primary_pdb": "/abs/path/to/primary.pdb",
    "ensemble_pdbs": ["/abs/path/to/decoy1.pdb", ...],
    "out_dir": "/abs/path/to/output",
    "checkpoint": "caliby",        // optional, default "caliby"
    "num_seqs": 1,                 // optional, default 1
    "fixed_pos_seq": "A10,A25-28", // optional
    "fixed_pos_override_seq": null // optional
}

Output (single-job):
{
    "seqs": ["MTYK..."],
    "energies": [-123.4],
    "primary_seq": "..."
}

(B) Batched multi-job mode (loads the Caliby model once and processes all
    primaries in one process — every primary still becomes its own
    ensemble_sample batch, so per-primary outputs are identical to running
    the legacy mode N times, minus subprocess + model-load overhead):
{
    "jobs": [
        {"key": "main_00",
         "primary_pdb": "/abs/path/...",
         "ensemble_pdbs": ["/abs/path/...", ...],
         "fixed_pos_seq": "A10,A25-28",          // optional
         "fixed_pos_override_seq": "A10:E,A25:D" // optional
        },
        ...
    ],
    "out_dir": "/abs/path/to/output",
    "checkpoint": "caliby",        // optional, default "caliby"
    "num_seqs": 1                  // optional, default 1
}

Output (batched):
{
    "results": [
        {"key": "main_00",
         "seqs": ["MTYK..."],
         "energies": [-123.4],
         "primary_seq": "..."},
        ...
    ]
}
"""
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _normalize_args(args):
    """Returns (jobs_list, is_batched). Legacy mode is converted to a 1-job list."""
    if "jobs" in args:
        return args["jobs"], True
    job = {
        "key": Path(args["primary_pdb"]).stem,
        "primary_pdb": args["primary_pdb"],
        "ensemble_pdbs": args["ensemble_pdbs"],
        "fixed_pos_seq": args.get("fixed_pos_seq"),
        "fixed_pos_override_seq": args.get("fixed_pos_override_seq"),
    }
    return [job], False


def main():
    args_path = sys.argv[1]
    with open(args_path) as f:
        args = json.load(f)

    # Lazy import so argparsing still works without caliby installed
    from caliby import clean_pdbs, load_model, make_constraints

    jobs, is_batched = _normalize_args(args)
    out_dir = args["out_dir"]
    checkpoint = args.get("checkpoint", "caliby")
    num_seqs = args.get("num_seqs", 1)
    temperature = args.get("temperature")  # forwarded to caliby ensemble_sample

    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        pdb_to_conformers = {}
        constraint_specs = {}

        for job in jobs:
            job_key = job["key"]
            primary_pdb = job["primary_pdb"]
            ensemble_pdbs = job["ensemble_pdbs"]
            # Clean primary first, then decoys — Caliby requires consistent residx/chain
            all_pdbs = [primary_pdb] + [
                p for p in ensemble_pdbs
                if os.path.abspath(p) != os.path.abspath(primary_pdb)
            ]
            # Clean into a per-job subdir so decoy filenames (e.g. decoy_00.pdb)
            # don't collide across jobs in batched mode.
            job_cleaned_dir = os.path.join(tmp, "cleaned", job_key)
            cleaned = clean_pdbs(all_pdbs, out_dir=job_cleaned_dir, num_workers=1)
            primary_cleaned = cleaned[0]

            # Canonicalize the primary stem to `job_key` so Caliby's example_id
            # (derived from Path(pdb_path).stem) matches the constraint_df row
            # and the result mapping below. Skips rename if already matching.
            renamed_primary = os.path.join(job_cleaned_dir, f"{job_key}.cif")
            if os.path.abspath(primary_cleaned) != os.path.abspath(renamed_primary):
                os.rename(primary_cleaned, renamed_primary)
                primary_cleaned = renamed_primary
            decoys_cleaned = cleaned[1:]

            pdb_to_conformers[job_key] = [primary_cleaned] + decoys_cleaned

            spec = {}
            if job.get("fixed_pos_seq"):
                spec["fixed_pos_seq"] = job["fixed_pos_seq"]
            if job.get("fixed_pos_override_seq"):
                spec["fixed_pos_override_seq"] = job["fixed_pos_override_seq"]
            if spec:
                constraint_specs[job_key] = spec

        constraint_df = None
        if constraint_specs:
            constraint_df = make_constraints(constraint_specs)

        model = load_model(checkpoint)
        kwargs = dict(num_seqs_per_pdb=num_seqs, out_dir=out_dir)
        if constraint_df is not None:
            kwargs["pos_constraint_df"] = constraint_df
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        results = model.ensemble_sample(pdb_to_conformers, **kwargs)

    # Re-group by example_id (== job_key after the rename above).
    seqs_by_id = defaultdict(list)
    energies_by_id = defaultdict(list)
    input_seq_by_id = {}
    for i, ex_id in enumerate(results["example_id"]):
        seqs_by_id[ex_id].append(results["seq"][i])
        energies_by_id[ex_id].append(float(results["U"][i]))
        if "input_seq" in results:
            input_seq_by_id[ex_id] = results["input_seq"][i]

    if is_batched:
        per_job = []
        for job in jobs:
            jk = job["key"]
            per_job.append({
                "key": jk,
                "seqs": seqs_by_id[jk],
                "energies": energies_by_id[jk],
                "primary_seq": input_seq_by_id.get(jk),
            })
        out = {"results": per_job}
    else:
        jk = jobs[0]["key"]
        out = {
            "seqs": seqs_by_id[jk],
            "energies": energies_by_id[jk],
            "primary_seq": input_seq_by_id.get(jk),
        }
    print("---CALIBY_RESULT_BEGIN---")
    print(json.dumps(out))
    print("---CALIBY_RESULT_END---")


if __name__ == "__main__":
    main()
