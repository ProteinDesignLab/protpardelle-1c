"""Build RF3 input JSONs from LigandMPNN outputs.

For each (sample, designed_seq) pair, emits one entry in a per-motif JSON:
  {
    "name": "<model>__<motif>__<sample_idx>__seq<j>",
    "components": [
      {"seq": "...", "chain_id": "A"},
      {"ccd_code": "<ligand1>"},
      {"ccd_code": "<ligand2>"},
      ...
    ]
  }

Outputs one JSON per (model, motif) so RF3 batch sizes are manageable. Names
are unique across the entire run so RF3 outputs can be linked back to the
original sample via a generated mapping CSV.

Usage:
  python scripts/eval_ame/prepare_rf3_inputs.py \
      --manifest <manifest.csv> \
      --lmpnn-dir <ligandmpnn_outputs> \
      --out-dir <rf3_inputs/>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_fasta(fp: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) tuples."""
    recs = []
    header, chunks = None, []
    with open(fp) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        recs.append((header, "".join(chunks)))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lmpnn-dir", required=True,
                    help="Top-level LigandMPNN output dir (has <model>/<motif>/<sample_idx>/seqs/*.fa)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-native", action="store_true",
                    help="Also include the native (input) sequence in the RF3 inputs")
    ap.add_argument("--max-per-json", type=int, default=200,
                    help="Max entries per output JSON file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Group entries by (model, motif) so each motif's RF3 run is self-contained
    by_motif: dict[tuple[str, str], list[dict]] = {}
    mapping_rows = []  # links RF3 entry name -> manifest row

    for row in rows:
        model = row["model_name"]
        motif = row["motif_name"]
        sidx = int(row["sample_idx"])
        sample_path = Path(row["sample_path"])
        ligands = [x for x in row["ligands"].split(",") if x.strip()]
        stoich = json.loads(row.get("ligand_stoichiometry_json", "{}")) if row.get("ligand_stoichiometry_json") else {}
        # Locate LMPNN output FASTA — LMPNN names it <sample_stem>.fa
        seqs_fa = (Path(args.lmpnn_dir) / model / motif / f"{sidx:02d}" /
                   "seqs" / f"{sample_path.stem}.fa")
        if not seqs_fa.exists():
            print(f"  WARN: missing LMPNN output {seqs_fa}")
            continue

        records = parse_fasta(seqs_fa)
        # First entry in LMPNN's FASTA is typically the input sequence
        # (label includes "T=0.0,..."); subsequent are designed.
        for j, (header, seq) in enumerate(records):
            is_native = (j == 0)
            if is_native and not args.include_native:
                continue
            seq = seq.replace("/", "").replace(":", "")  # strip chain separators
            if not seq:
                continue
            entry_name = f"{model}__{motif}__{sidx:02d}__seq{j}"
            components = [{"seq": seq, "chain_id": "A"}]
            for ligand in ligands:
                count = int(stoich.get(ligand.upper(), 1))
                for _ in range(max(1, count)):
                    components.append({"ccd_code": ligand})
            entry = {"name": entry_name, "components": components}
            by_motif.setdefault((model, motif), []).append(entry)
            mapping_rows.append({
                "rf3_name": entry_name,
                "model_name": model,
                "motif_name": motif,
                "sample_idx": sidx,
                "lmpnn_seq_idx": j,
                "sequence": seq,
                "sample_path": str(sample_path),
                "scaffold_motif_resnums": row["scaffold_motif_resnums"],
                "contig_order": row["contig_order"],
                "ligands": row["ligands"],
                "motif_pdb_path": row["motif_pdb_path"],
                "conditioning_atoms_json": row["conditioning_atoms_json"],
            })

    # Write per-motif JSONs (chunked by --max-per-json)
    json_paths = []
    for (model, motif), entries in by_motif.items():
        for chunk_i in range(0, len(entries), args.max_per_json):
            chunk = entries[chunk_i:chunk_i + args.max_per_json]
            fname = f"rf3_input__{model}__{motif}__chunk{chunk_i // args.max_per_json:02d}.json"
            out_fp = out_dir / fname
            with open(out_fp, "w") as f:
                json.dump(chunk, f, indent=2)
            json_paths.append(str(out_fp))

    # Write mapping CSV — maps RF3 entry name back to manifest row
    mapping_fp = out_dir / "rf3_mapping.csv"
    if mapping_rows:
        with open(mapping_fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=mapping_rows[0].keys())
            w.writeheader()
            w.writerows(mapping_rows)

    # Write a manifest of JSON paths for sbatch chunking
    json_list_fp = out_dir / "rf3_input_jsons.txt"
    with open(json_list_fp, "w") as f:
        for p in json_paths:
            f.write(p + "\n")

    print(f"wrote {len(json_paths)} input JSONs (total {len(mapping_rows)} entries)")
    print(f"  per-motif JSONs:   {out_dir}/rf3_input__*.json")
    print(f"  mapping CSV:       {mapping_fp}")
    print(f"  json list (txt):   {json_list_fp}")


if __name__ == "__main__":
    main()
