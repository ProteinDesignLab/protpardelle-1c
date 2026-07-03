# CLAUDE.md - Protpardelle Codebase Guide

## Overview

Protpardelle is a protein structure diffusion model for backbone and all-atom protein generation with motif scaffolding capabilities. It supports conditional generation where specific structural motifs are preserved while generating novel scaffold regions.

There are two parallel architectures:
1. **Atom37** (original): Fixed 37-atom-per-residue representation, DiT/U-ViT denoiser
2. **Per-atom** (new): Hierarchical atom transformer, each atom is a token `[B, T, 3]` where `T = n_groups × max_atoms_per_group (14)`. Supports metals and arbitrary ligands.

## Project Structure

```
protpardelle-1c/
├── src/protpardelle/
│   ├── core/
│   │   ├── models.py          # Protpardelle class, MiniMPNN, MiniCaliby, CoordinateDenoiser, AtomCoordinateDenoiser
│   │   ├── modules.py         # Network building blocks: HierarchicalAtomTransformer, GroupGraphMPNN, NoiseConditioningBlock, etc.
│   │   └── diffusion.py       # Noise schedules, noise_coords, noise_coords_per_atom
│   ├── data/
│   │   ├── dataset.py         # PDBDataset (atom37), PerAtomPDBDataset, StochasticMixedSampler, calc_sigma_data
│   │   ├── pdb_io.py          # load_feats_from_pdb, load_feats_from_pdb_per_atom, feats_to_pdb_str_per_atom
│   │   ├── atom.py            # Per-atom featurization: pdb_to_per_atom_features, pad/crop utilities
│   │   └── motif.py           # Motif parsing utilities (contig_to_motif_placement, etc.)
│   ├── common/
│   │   └── residue_constants.py  # Atom names, element vocab, group type mappings, MAX_ATOMS_PER_GROUP=14
│   ├── train.py               # ProtpardelleRunner, compute_loss, compute_loss_per_atom, train()
│   └── sample.py              # Entry point: sample() for atom37, sample_per_atom() for per-atom
├── examples/
│   ├── sampling/              # YAML configs for sampling runs
│   ├── motifs/                # Motif PDB files organized by project
│   └── training/              # Training configs (cc91-peratom*.yaml for per-atom)
└── scripts/
    ├── train.sbatch           # 1-GPU training sbatch
    └── train_2gpu.sbatch      # 2-GPU DataParallel training sbatch
```

## Per-Atom Architecture

### Tokenization
- Each atom is one token: coordinates `[B, T, 3]` where `T = n_groups × 14`
- Atoms within a group are contiguous: group 0 occupies indices 0-13, group 1 occupies 14-27, etc.
- Element vocabulary: 128 entries indexed by atomic number (H=1, C=6, N=7, O=8, Fe=26, Sm=62)
- Group types: 0-19 = amino acids, 20 = UNK, 25 = metal, 26 = water

### Key tensors (per-atom format)
- `atom_positions`: `[B, T, 3]` — all atom coordinates
- `atom_mask`: `[B, T]` — valid atom mask (padding atoms = 0)
- `atom_element`: `[B, T]` — atomic number per atom
- `group_id`: `[B, T]` — which group each atom belongs to
- `group_mask`: `[B, G]` — valid group mask
- `group_type`: `[B, G]` — amino acid / metal type per group
- `atom_name_id`: `[B, T]` — atom index within group (0=N, 1=CA, 2=C, 3=O for protein backbone)

### HierarchicalAtomTransformer (modules.py)
Architecture: local attention (window=48) → group pool (scatter_mean) → global attention (on G groups) → group distribute → local attention, repeated L times. Uses Karras preconditioning with `sigma_data`.

### Model Classes (models.py)

**Protpardelle** — Main wrapper class:
- `forward()` dispatches to `forward_atom37()` or `forward_per_atom()` based on `self.is_per_atom`
- `forward_per_atom()` returns 4-tuple: `(denoised_coords, struct_self_cond, group_type_logprobs, seq_self_cond)`
- `sample_per_atom()` — ODE sampling loop for per-atom models
- Supports DataParallel for multi-GPU training

**MiniCaliby** (~1.9M params) — Group-level identity prediction (analogous to MiniMPNN for atom37):
- KNN graph MPNN on group center coordinates
- RBF edge features, encoder-decoder message passing with noise conditioning
- Input: denoised per-atom coords → Output: group_type log-probabilities `[B, G, V]`
- Operates on `denoised_coords.detach()` (no gradient to struct model)
- Self-conditioning: feeds back previous group_type predictions

**MiniMPNN** — Sequence prediction for atom37 models (wraps ProteinMPNN)

### Training Tasks
- `"ai-allatom-nomask"`: Structure denoising (both atom37 and per-atom)
- `"backbone"`, `"allatom"`: Atom37-only structure tasks
- `"seqdes"`: Train MiniMPNN with frozen struct model
- `"groupdes"`: Train MiniCaliby with frozen per-atom struct model
- `"codesign"` / `"groupcodesign"`: Joint training of struct + sequence/identity models

### Training Configs (per-atom)
- `cc91-peratom.yaml` — Rhaven overfitting (fixed_size=128, batch_size=32, include_metals=True)
- `cc91-peratom-cath.yaml` — CATH training (fixed_size=256, batch_size=32, use_amp=True)
- `cc91-peratom-caliby.yaml` — MiniCaliby training (groupdes task, frozen struct model)

## Atom37 Architecture (Original)

### Entry Point: `src/protpardelle/sample.py`

The `sample()` function is the main entry point for atom37 models:
```bash
python -m protpardelle.sample <config.yaml> --motif-dir <path> --num-samples N
```

**Flow:**
1. Load runner config YAML with `motifs`, `motif_contigs`, `total_lengths`, `hotspots`, `ssadj`
2. For each motif in config, resolve path and parse contigs
3. Call `generate()` → `draw_samples()` → `model.sample()`

### Motif Conditioning

**Contig format:** `"20-80;A58-71;10-50;A96-96;70-150"`
- Numbers = scaffold regions with length ranges
- Letters+numbers = motif segments (e.g., `A58-71` = chain A residues 58-71)

**Crop conditional guidance** (`apply_crop_cond_strategy`):
- `"backbone"`: Only backbone atoms (N, CA, C, O)
- `"sidechain"`: Sidechain atoms
- `"sidechain-tip"`: Only tip atoms of sidechains

### Multi-Motif Conditioning

Supports weighted averaging of multiple motifs:
```yaml
motifs:
  - [["pre_cyclic", 1], ["post_cyclic", 1]]  # Equal weight average
  - [["state_A", 5], ["state_B", 3]]         # Weighted toward state_A
```

### Key Data Structures (atom37)

**motif_feats** (from `load_feats_from_pdb`):
- `atom_positions`: `[num_res, 37, 3]` all-atom coordinates
- `atom_mask`: `[num_res, 37]` valid atom mask
- `aatype`: `[num_res]` amino acid indices
- `residue_index`: `[num_res]` residue numbers
- `chain_index`: `[num_res]` chain indices

**Atom indexing** (37-atom representation):
- 0: N, 1: CA, 2: C, 3: CB, 4: O, 5+: sidechain atoms

## Sampling Config YAML Format (atom37)

```yaml
search_space:
  models: [["model_name", "epoch", "sampling_config"]]
  step_scales: [1.0, 1.2]
  schurns: [0.0, 200]
  crop_cond_starts: [0.0]
  translations: [[0.0, 0.0, 0.0]]

motifs:
  - motif_name              # Single motif
  - [["name1", w1], ...]    # Multi-motif with weights

motif_contigs:
  - "20-80;A58-71;10-50"    # Contig string per motif entry

total_lengths:
  - [[128, 256]]            # Length range per chain

hotspots: [null]            # Optional hotspot residues
ssadj: [null]               # Optional secondary structure adjacency
```

## Common Commands

```bash
# Atom37 sampling with motif scaffolding
python -m protpardelle.sample examples/sampling/config.yaml \
    --motif-dir examples/motifs/project \
    --num-samples 1000 \
    --num-mpnn-seqs 8

# Per-atom sampling (unconditional)
python -m protpardelle.sample per-atom \
    /oak/stanford/groups/possu/tianyu/models/protpardelle-1c/configs/cc91-peratom-ov.yaml \
    /oak/stanford/groups/possu/tianyu/models/protpardelle-1c/weights/cc91-peratom-ov_epoch439.pth \
    --output-dir /scratch/users/tianyulu/protpardelle-1c-sampling-results/my_run \
    --num-samples 8 --n-steps 200 --step-scale 1.0 --s-churn 0

# Per-atom sampling with motif conditioning
python -m protpardelle.sample per-atom \
    <config.yaml> <checkpoint.pth> \
    --motif-pdb examples/motifs/lanm/8FNS.pdb \
    --output-dir /scratch/users/tianyulu/protpardelle-1c-sampling-results/my_run \
    --num-samples 8

# Training (1 GPU)
sbatch --job-name=cc91-peratom-cath scripts/train.sbatch cc91-peratom-cath /scratch/users/tianyulu/protpardelle-1c-experiments

# Training (2 GPU DataParallel)
sbatch --job-name=cc91-peratom-cath scripts/train_2gpu.sbatch cc91-peratom-cath /scratch/users/tianyulu/protpardelle-1c-experiments
```

## Important Paths

- **Model weights & configs**: `/oak/stanford/groups/possu/tianyu/models/protpardelle-1c/{weights,configs}/`
  - Naming: `{name}_epoch{N}.pth` for weights, `{name}.yaml` for configs
  - `PROTPARDELLE_MODEL_PARAMS` env var points here (used by atom37 sample command)
- **Training checkpoints**: `/scratch/users/tianyulu/protpardelle-1c-experiments/protpardelle-1c/{exp_name}/checkpoints/`
  - `epoch{N}_model.pth` — full model object (torch.save(model))
  - `epoch{N}_training_state.pth` — dict with `model_state_dict`, `optim_state_dict`, `epoch`, `total_steps`
  - Both formats are supported by `load_model()` and `load_pretrained_module()`
- **Sampling results**: `/scratch/users/tianyulu/protpardelle-1c-sampling-results/`
- **Datasets**:
  - Rhaven: `/scratch/users/tianyulu/datasets/rhaven` (20 metalloprotein PDBs, for overfitting)
  - CATH: `/scratch/users/tianyulu/augmented_ingraham_cath_bugfree` (337k designable_ PDBs)
- **Slurm logs**: `/scratch/users/tianyulu/slurm_log_files/`

## Performance Notes

- Per-atom with fixed_size=256 → T=3584 tokens. FP32 batch_size=16 uses ~47GB on A100-80GB.
- AMP (use_amp=True) gives ~2.5x speedup and allows batch_size=32 (55GB peak).
- Data loading with num_workers=0 is 0.45s/batch; num_workers=4+ reduces to 0.16s/batch.
- Self-conditioning (90% prob) roughly doubles compute per step (2 forward passes).
- DataParallel supported via `--use-dataparallel` flag in train CLI and `forward()` dispatch.

## Environment Rules

- **NEVER use `pip install` directly** — it defaults to `~/.local` which fills the 15GB home directory. Always use `uv pip install` with `VIRTUAL_ENV=<path>` set, or use the venv's own pip.
- **Always set `TMPDIR=/scratch/users/tianyulu/tmp`** before pip/uv operations to avoid build temp files filling home.
- The `vtx` PyPI package provides the `vortex` Python module (StripedHyena).

## Development Notes

- `forward()` dispatches to `forward_atom37()` or `forward_per_atom()` based on `self.is_per_atom`
- Per-atom dataset uses `PerAtomPDBDataset`; atom37 uses `PDBDataset`
- `calc_sigma_data` auto-detects per-atom vs atom37 format via `"atom_positions" in inputs`
- Overfit mode: `--overfit N` tiles dataset to at least N samples (fixed tiling logic in PerAtomPDBDataset)
- Checkpoint loading handles both full-model and state-dict formats throughout the codebase
