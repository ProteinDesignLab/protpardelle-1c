# AME design + eval pipeline

End-to-end evaluation of protpardelle-1c samples for the AME (M-CSA active-site)
benchmark:

```
sampling output (ame_*.yaml runs)
        │
        ▼   gather_samples.py
   manifest.csv                          ← scaffold positions of motif residues
        │
        ▼   run_ligandmpnn.sbatch
  designed sequences (8 / backbone)      ← motif residues fixed
        │
        ▼   prepare_rf3_inputs.py
   RF3 input JSONs                       ← seq + ligand CCD codes from ame.csv
        │
        ▼   run_rf3_dispatch.sbatch (calls launch_rf3_sbatch_apptainer.sh)
   RF3 predicted structures (.pdb / .cif)
        │
        ▼   compute_metrics.py
   per-sample metrics csv                ← scRMSD, motif scRMSD, ligand clashes
        │
        ▼   aggregate_results.py
   per-model / per-motif success rates
```

## Success criteria (all three must hold)
1. **scRMSD ≤ 2 Å** between protpardelle sample and RF3 prediction (CA, Kabsch-aligned)
2. **Motif scRMSD ≤ 1.5 Å** for the catalytic functional-group atoms listed in `ame.csv`
3. **No ligand clashes** — zero binder atoms within 1.5 Å of any RF3-predicted ligand atom

## Prerequisites
- protpardelle env: `/scratch/users/tianyulu/envs/protpardelle/`
- LigandMPNN env (or fall back to protpardelle env): `/scratch/users/tianyulu/envs/ligandmpnn/`
- RF3 env: `/scratch/users/tianyulu/envs/rf3/` (used inside the apptainer container)
- Apptainer image: `$GROUP_HOME/containers/pytorch_24.12.sif`

## Step 1 — Build sample manifest (one CSV per yaml)
After protpardelle sampling finishes, walk each output dir to record sample paths and
motif scaffold positions:

```bash
SAMP=/scratch/users/tianyulu/protpardelle-1c-sampling-results
EVAL=/scratch/users/tianyulu/ame_eval
MOTIF_DIR=/home/users/tianyulu/code/protpardelle-1c/examples/motifs/ame/motifs
CSV=/home/users/tianyulu/code/protpardelle-1c/examples/motifs/ame/ame.csv
mkdir -p ${EVAL}

YAML_DIR=/home/users/tianyulu/code/protpardelle-1c/examples/sampling/ame
for yaml in cc91_tip_condmask cc91_tip_condmask_large cc138 cc137; do
    python scripts/eval_ame/gather_samples.py \
        --sample-dir ${SAMP}/ame_${yaml} \
        --motif-dir ${MOTIF_DIR} \
        --csv ${CSV} \
        --yaml ${YAML_DIR}/ame_${yaml}.yaml \
        --out ${EVAL}/manifest__${yaml}.csv
done

# (optional) merge into one manifest
head -n 1 ${EVAL}/manifest__cc138.csv > ${EVAL}/manifest_all.csv
for f in ${EVAL}/manifest__*.csv; do tail -n +2 "$f" >> ${EVAL}/manifest_all.csv; done
```

The manifest captures **scaffold_motif_resnums** (the 1-indexed PDB residue
numbers in the sample where motif residues were placed) and **contig_order**
(the original residue numbers in contig order). compute_metrics.py uses these
to compute motif scRMSD without re-parsing scaffold_info.csv.

## Step 2 — LigandMPNN sequence design (8 / backbone)
```bash
sbatch scripts/eval_ame/run_ligandmpnn.sbatch \
    ${EVAL}/manifest_all.csv \
    ${EVAL}/lmpnn_outputs \
    8 8        # num_seqs, batch_size
```

Outputs at `${EVAL}/lmpnn_outputs/<model>/<motif>/<sample_idx>/seqs/<sample>.fa`.
Motif residues from the manifest are fixed via `--fixed_residues "A13 A26 ..."`.

## Step 3 — Build RF3 input JSONs
```bash
python scripts/eval_ame/prepare_rf3_inputs.py \
    --manifest ${EVAL}/manifest_all.csv \
    --lmpnn-dir ${EVAL}/lmpnn_outputs \
    --out-dir ${EVAL}/rf3_inputs
```

Produces:
- `rf3_input__<model>__<motif>__chunkNN.json` — per-motif chunks (default ≤200 entries each)
- `rf3_mapping.csv` — links every RF3 entry name back to manifest row + sequence index
- `rf3_input_jsons.txt` — list of JSON paths for sbatch chunking

## Step 4 — Predict structures with RF3 (apptainer)
```bash
N=$(($(wc -l < ${EVAL}/rf3_inputs/rf3_input_jsons.txt) - 1))
# Note: use launch_rf3_sbatch_apptainer.sh on the dispatch wrapper so each
# array task runs inside the rf3 container.
bash /home/users/tianyulu/code/rhaven/scripts/structure_prediction/rf3/launch_rf3_sbatch_apptainer.sh \
    scripts/eval_ame/run_rf3_dispatch.sbatch
# (the launcher's wrapper will preserve --array=0-${N}%4 if you set it on the dispatch sbatch)
```

Or simpler — submit per-JSON manually:
```bash
while read JSON; do
    sbatch --export=JSON_INPUT=${JSON} scripts/eval_ame/run_rf3.sbatch ${JSON}
done < ${EVAL}/rf3_inputs/rf3_input_jsons.txt
```

RF3 outputs land in its configured output dir (`/scratch/users/tianyulu/modelforge/...`
unless overridden). Note the directory in which the predicted PDB/CIF files appear
— pass it as `--rf3-out-dir` to the next step.

## Step 5 — Compute metrics
```bash
python scripts/eval_ame/compute_metrics.py \
    --mapping ${EVAL}/rf3_inputs/rf3_mapping.csv \
    --rf3-out-dir <RF3_OUTPUT_DIR_FROM_STEP_4> \
    --out ${EVAL}/metrics.csv
```

For each (sample, designed-seq) pair, computes scRMSD, motif scRMSD, ligand-clash
count and a binary `success` column.

## Step 6 — Aggregate
```bash
python scripts/eval_ame/aggregate_results.py \
    --metrics ${EVAL}/metrics.csv \
    --out-dir ${EVAL}/summary/
```

Prints global pass rate; writes `summary/by_model.csv`, `by_motif.csv`,
`by_model_motif.csv`.

## Files in this directory

| File | Purpose |
|---|---|
| `gather_samples.py` | Walk sample dirs → manifest CSV with scaffold motif positions |
| `run_ligandmpnn.sbatch` + `_run_ligandmpnn_chunk.py` | LigandMPNN seq design with fixed motif residues |
| `prepare_rf3_inputs.py` | FASTA → RF3 input JSONs + mapping CSV |
| `run_rf3.sbatch` | Inner RF3 sbatch (called inside apptainer) |
| `run_rf3_dispatch.sbatch` | Outer array dispatch picking one JSON per task |
| `compute_metrics.py` | Per-sample scRMSD / motif scRMSD / clash metrics |
| `aggregate_results.py` | Per-model / per-motif success rate summaries |

## Notes on motif index tracking

- For **indexed sampling** (cc91-tip-condmask, cc91-tip-condmask-large, cc138):
  `scaffold_info.csv::motif_placements` is a string like `12/A/12/A/9/A/16/A/27`
  meaning 12 scaffold + motif-residue + 12 + ... `gather_samples.py` parses this
  into 0-indexed positions and converts to 1-indexed PDB resnums.
- For **unindexed sampling** (cc137): `scaffold_info.csv::matched_motif_idx` already
  contains 1-indexed PDB resnums (from linear_sum_assignment).
- The manifest's `contig_order` records the original motif residue identifiers
  (e.g. `A215,A33,A60,A62,A65,A123` for M0050 singlechain) so motif RMSD pairs
  the correct atoms at the correct scaffold positions.
- Multi-chain motifs (M0050, M0058, M0093, M0375, M0732) were collapsed into
  single chain A in `*_specific_atoms_singlechain.pdb` files. The original chain
  letter is recoverable only via the `conditioning_atoms_json` column (which
  uses the original `<chain><resnum>` keys from `ame.csv`).
