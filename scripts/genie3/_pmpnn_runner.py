"""ProteinMPNN sequence design runner.

Invoked as a subprocess by ``af2.py`` so it runs in the ligandmpnn uv env. Calls
LigandMPNN's ``run.py`` in protein_mpnn mode to redesign the binder chain only.

Reads a JSON job spec from --jobs, writes a JSON result to --out.

Job spec schema::

    {
      "samples": [
        {"pdb": "/path/to/sample.pdb",
         "key": "sample_01_bhrf1_00",
         "binder_chain": "B"},
        ...
      ],
      "num_seqs": 8,
      "temperature": 0.1,
      "seed": 42,
      "out_dir": "/path/to/pmpnn_out",
      "ligandmpnn_repo": "/home/users/tianyulu/code/LigandMPNN",
      "checkpoint": "/oak/.../v_48_020.pt"
    }

Result schema::

    {"results": [{"key": ..., "binder_seqs": ["AAA...", ...], "fa_path": "..."}]}
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_fa(fa_path):
    """Read a LigandMPNN .fa output. Returns list of designed sequences (skips native).

    Each sequence is the full ``chainA:chainB:...`` string.
    """
    seqs = []
    with open(fa_path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    first = True
    while i < len(lines):
        if lines[i].startswith(">"):
            seq = lines[i + 1] if i + 1 < len(lines) else ""
            if not first:
                seqs.append(seq)
            first = False
            i += 2
        else:
            i += 1
    return seqs


def extract_binder_seq(full_seq, pdb_path, binder_chain):
    """Split a ProteinMPNN sequence (``chainA:chainB:...``) and pull the binder chain.

    Chains in the .fa output are in the order they appear in the input PDB ATOM
    records, separated by ':'.
    """
    chain_order = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                if ch not in chain_order:
                    chain_order.append(ch)
    pieces = full_seq.split(":")
    if len(pieces) != len(chain_order):
        # Fallback: assume last piece is binder
        return pieces[-1]
    for ch, piece in zip(chain_order, pieces):
        if ch == binder_chain:
            return piece
    return pieces[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.jobs) as f:
        job = json.load(f)

    samples = job["samples"]
    num_seqs = int(job.get("num_seqs", 8))
    temperature = float(job.get("temperature", 0.1))
    seed = int(job.get("seed", 42))
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = job["ligandmpnn_repo"]
    ckpt = job["checkpoint"]

    results = []
    for s in samples:
        sample_pdb = s["pdb"]
        binder_chain = s["binder_chain"]
        sub_out = out_dir / s["key"]
        sub_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            f"{repo}/run.py",
            "--model_type", "protein_mpnn",
            "--checkpoint_protein_mpnn", ckpt,
            "--pdb_path", sample_pdb,
            "--out_folder", str(sub_out),
            "--chains_to_design", binder_chain,
            "--temperature", str(temperature),
            "--seed", str(seed),
            "--number_of_batches", "1",
            "--batch_size", str(num_seqs),
            "--save_stats", "0",
            "--verbose", "0",
        ]
        subprocess.run(cmd, check=True, cwd=repo)
        fa_path = sub_out / "seqs" / f"{Path(sample_pdb).stem}.fa"
        if not fa_path.exists():
            raise RuntimeError(f"ProteinMPNN did not produce {fa_path}")
        full_seqs = parse_fa(fa_path)
        binder_seqs = [extract_binder_seq(fs, sample_pdb, binder_chain) for fs in full_seqs]
        results.append({"key": s["key"], "pdb": sample_pdb, "binder_seqs": binder_seqs, "fa_path": str(fa_path)})

    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
