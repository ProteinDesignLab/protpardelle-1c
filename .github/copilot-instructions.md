# Protpardelle-1c AI Assistant Instructions

Concise, project-specific guidance to help an AI coding agent be productive quickly. Focus on THESE repo conventions (not generic ML advice).

## 1. High-level Architecture

Protpardelle-1c is a diffusion-based protein structure (and sequence) generative framework.

Main layers:

- `src/protpardelle/core/` - Core modeling logic
  - `models.py` defines composite model class `Protpardelle` plus submodules: `CoordinateDenoiser` (U-ViT/DiT style) and optional `MiniMPNN` for sequence scoring / design; orchestrates conditioning (motifs, hotspots, sidechain/backbone crop conditions) and sampling loops.
  - `diffusion.py` noise schedules + coordinate perturbation utilities. Noise level increases with timestep; keep this monotonic increasing convention consistent when adding schedules.
  - `modules.py` contains reusable NN building blocks (attention, resblocks, positional encodings incl. chain-relative and rotary variants) and learning rate scheduler `LinearWarmupCosineDecay`.
- `src/protpardelle/data/` - Data processing and I/O abstractions
  - `pdb_io.py` feature extraction from PDB/mmCIF -> tensors and PDB writers (`load_feats_from_pdb`, `write_coords_to_pdb`). Respect `chain_residx_gap` logic when generating multi-chain residues.
  - `dataset.py` dataset + batching utilities (`PDBDataset`, cropping / random rotations, `make_fixed_size_1d`).
  - `motif.py` parses contig spec strings (RFdiffusion-like) into motif placement indices.
  - `atom.py` masks and conversions between backbone-only and full atom (37) representations.
  - `align.py` alignment functions including Kabsch algorithm for structural alignment.
  - `cycpep.py` cyclic peptide utilities for handling cyclic protein structures.
  - `sequence.py` sequence-related utilities for amino acid type conversions.
- `src/protpardelle/integrations/` - External model integrations
  - `esmfold.py` ESMFold integration for structure evaluation.
  - `protein_mpnn.py` ProteinMPNN sequence design integration - loaded lazily based on flags (`num_mpnn_seqs`).
- `src/protpardelle/common/` - Shared constants and data structures
  - `residue_constants.py` static biochemical constants and atom ordering.
  - `protein.py` lightweight protein data containers (`Protein`, `Hetero`, `PDB_CHAIN_IDS`) reused across I/O, sampling, training and PDB serialization (`to_pdb`).
- `src/protpardelle/configs/` - Configuration management
  - `running_dataclasses.py`, `sampling_dataclasses.py`, `training_dataclasses.py` - typed configuration classes.
  - `running/` directory contains runtime configuration files.
- `src/protpardelle/` - Main entry points and utilities
  - `sample.py` - CLI (Typer) sampling orchestrator; builds search space combos, runs model sampling, optional ProteinMPNN sequence design, evaluation (self-consistency) and writes results under `results/` (or `PROTPARDELLE_OUTPUT_DIR`).
  - `train.py` - CLI training loop with mixed precision + (optionally) `nn.DataParallel`.
  - `evaluate.py` - Sequence design and self-consistency utilities.
  - `likelihood.py` - Likelihood computation entry point for model evaluation.
  - `env.py` - Environment variables and path definitions with auto-detection.
  - `utils.py` - Shared utility functions for seeding, device selection, path normalization.

## 2. Environment Setup

**CRITICAL: Always activate the conda environment before running any Python commands.**

```bash
conda activate protpardelle
```

This environment contains all required dependencies and must be activated before:

- Running sampling: `python -m protpardelle.sample`
- Running training: `python -m protpardelle.train`
- Running likelihood computation: `python -m protpardelle.likelihood`
- Running evaluation: `python -m protpardelle.evaluate`
- Running any tests or development scripts

The conda environment should be set up according to the project's dependency requirements (see `pyproject.toml` or environment files in the repository).

## 3. Configuration & Execution

- Configuration is managed through typed dataclasses in `src/protpardelle/configs/`:
  - `running_dataclasses.py` - Runtime configuration classes
  - `sampling_dataclasses.py` - Sampling-specific configuration
  - `training_dataclasses.py` - Training-specific configuration
  - `running/` directory contains runtime configuration files
- Configs live under `examples/sampling/*.yaml` and `examples/training/*.yaml`. They are not Hydra multilevel packages; sampling builds Cartesian products over `search_space` lists manually (see `sample.py`). Maintain parameter names exactly (e.g. `step_scales`, `schurns`, `crop_cond_starts`).
- Environment variables (see `env.py`) override auto-detected paths: `PROTPARDELLE_MODEL_PARAMS`, `PROTEINMPNN_WEIGHTS`, `ESMFOLD_PATH`, `PROTPARDELLE_OUTPUT_DIR`, `FOLDSEEK_BIN`, `PROJECT_ROOT_DIR`.
- Model checkpoints + configs stored under `model_params/` (subfolders `configs/` + `weights/`). Loading expects a config name matching weight stems (e.g. `cc58_epoch416.pth` with `configs/cc58.yaml`).
- Path detection: `env.py` provides auto-detection of project root via `PROJECT_ROOT_DIR` env var, `pyproject.toml` search, or git root.

## 4. Sampling Workflow (critical path)

1. Read motif / input structure (if any) via `load_feats_from_pdb` -> features dict.
2. Build search space (product of user lists) -> each combination calls model inference.
3. Diffusion loop uses noise schedule in `diffusion.py`; `schurn` injects extra stochasticity (scaled internally by step count).
4. Crop / motif / hotspot conditioning implemented via coordinate masking (see `apply_crop_cond_strategy` in `models.py`). Sidechain tip conditioning uses curated atom lists in `residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS`.
5. Optional sequence design: ProteinMPNN (`integrations/protein_mpnn.py`) or MiniMPNN head (sequence diffusion / self-conditioning modes `seqdes`).
6. Write PDBs: for backbone-only samples first convert with `bb_coords_to_atom37_coords` if needed; chain IDs resolved either by provided mapping or sequential from `'A'` upward.

Key invariants: ensure `residue_index` starts at 1 post normalization; apply `add_chain_gap` only once; keep shape conventions `(B, N, A, 3)` for coords, `(B, N)` for indices/masks.

### 4.1. Likelihood Computation

- Entry point: `python -m protpardelle.likelihood` for model evaluation and likelihood computation.
- Uses same model loading and data processing as sampling but computes likelihood scores.
- Supports both backbone-only and full-atom likelihood computation.

## 5. Training Workflow

- Entry: `python -m protpardelle.train <model_name> <output_dir>` (often submitted via `scripts/train.sbatch`).
- Datasets assembled from YAML config fields referencing prepared AI-CATH / interface data (see README dataset section). Sampling noise schedule functions: `uniform`, `lognormal`, `mpnn`, `constant` (must match those in `diffusion.noise_schedule`).
- Losses: masked MSE for coordinates (`masked_mse_loss`) + optionally cross-entropy for sequence tokens (`masked_cross_entropy_loss`). When adding a new loss term, wire it in inside the training loop where `total_loss` is accumulated (search for existing accumulation pattern). Maintain masking semantics (divide by sum(mask) with clamp >=1e-6).

## 6. Data & Tensor Conventions

- Atom ordering fixed by `residue_constants.atom_order`; backbone indices stored in `bb_idxs` in `CoordinateDenoiser`.
- Chain indices are integer (0-based) while chain IDs in PDB are letters. The mapping is reversible via `chain_id_mapping` passed into writers.
- Masking: `atom_mask` shape `(N, 37)`; when absent, reconstructed from aatype (`atom37_mask_from_aatype`). Sequence mask denotes which residues are designable / present.
- Alignment: `align.py` provides Kabsch algorithm for structural alignment between coordinate sets.
- Cyclic peptides: `cycpep.py` handles cyclic protein structures with specialized utilities for bond formation and ring closure.
- Sequence processing: `sequence.py` provides utilities for amino acid type conversions with support for different token counts (20, 21, 22).

## 7. Extending the Model

- To add a new conditioning modality (e.g., distance map):
  1. Extend feature construction (likely in `sample.py` and training dataset assembly) producing a tensor aligned with `(B, N, A, 3)` or `(B, N, F)`.
  2. Modify `CoordinateDenoiser` input channel calculation (`nc_in`) ensuring ordering appended after existing xyz/selfcond/hotspot/ssadj channels.
  3. Update forward pass to concat new conditioning before projection into transformer / U-ViT blocks.
- To add a new noise schedule: implement in `diffusion.noise_schedule` with a new literal name; update any config enumerations and validation.

## 8. Common Pitfalls

- Forgetting to normalize `residue_index` (must start at 1) causes relative positional encoding mismatches.
- Applying `add_chain_gap` twice inflates residue indices and breaks relative chain encodings (search for single call in `load_feats_from_pdb`).
- Incorrect atom name introduces silent skip during PDB load (`if np.sum(mask) < 0.5: continue`). Provide canonical atoms or they’re dropped.
- When creating PDBs for backbone-only generation: ensure 4 atoms ordering (`N, CA, C, O`) else `bb_coords_to_pdb_str` may mis-assign residue boundaries.

## 9. Testing & Minimal Verification

- Test structure: `tests/` directory contains comprehensive test suite:
  - `test_likelihood.py` - Likelihood computation tests
  - `test_evaluate.py` - Evaluation and self-consistency tests
  - `test_env.py` - Environment and configuration tests
  - `conftest.py` - Shared test fixtures and configuration
  - `pytest.ini` - Test configuration and settings
- For quick smoke test after changes: load a small model config (e.g. `cc58`), run a single sample with `--num-samples 1 --num-mpnn-seqs 0` and confirm PDB output + no NaNs.
- Integration tests: Use synthetic coordinates and motif indices following tensor shape conventions `(B, N, A, 3)` for coords, `(B, N)` for indices/masks.

## 10. Style & Dependencies

- Python 3.12 / `uv` for dependency pinning; avoid adding large new deps unless essential. Use existing utilities in `utils.py` for seeding, device selection, path normalization.
- Logging via stdlib `logging`; prefer `logger.info` / `logger.warning` over print.

## 11. When Unsure

Reference: `models.py` for forward & sampling loops; `sample.py` for end-to-end pipeline; `pdb_io.py` for I/O conventions. Mirror existing patterns before refactoring. Ask for clarification if introducing API-breaking changes to CLI flags or config keys.

Key reference files:

- `core/models.py` - Model architecture and sampling loops
- `sample.py` - End-to-end sampling pipeline
- `data/pdb_io.py` - I/O conventions and PDB handling
- `data/atom.py` - Atom representation conversions
- `data/align.py` - Structural alignment utilities
- `data/cycpep.py` - Cyclic peptide handling
- `data/sequence.py` - Sequence processing utilities
- `likelihood.py` - Likelihood computation
- `env.py` - Environment and path management
- `utils.py` - Shared utility functions

## 12. ASCII only for generated code

- All generated code (source, diffs, patches) must use plain ASCII characters.
- Allowed: standard punctuation, quotes ' " , parentheses, brackets, braces, hash for comments, backticks for Markdown.
- Disallowed in code: Unicode arrows, en/em dashes, smart quotes, multiplication dot, fancy minus, ellipsis character, box drawing characters, mathematical symbols.
- Replace stylistic arrows with `->`, long dashes with `-`, ellipsis with `...`.
- Documentation sections may use Markdown backticks but avoid decorative Unicode for consistency.
