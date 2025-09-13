## Protpardelle-1c AI Assistant Instructions

Concise, project-specific guidance to help an AI coding agent be productive quickly. Focus on THESE repo conventions (not generic ML advice).

### 1. High-level Architecture

Protpardelle-1c is a diffusion-based protein structure (and sequence) generative framework.

Main layers:

- `src/protpardelle/core/` - Modeling logic
  - `models.py` defines composite model class `Protpardelle` plus submodules: `CoordinateDenoiser` (U-ViT/DiT style) and optional `MiniMPNN` for sequence scoring / design; orchestrates conditioning (motifs, hotspots, sidechain/backbone crop conditions) and sampling loops.
  - `diffusion.py` noise schedules + coordinate perturbation utilities. Noise level increases with timestep; keep this monotonic increasing convention consistent when adding schedules.
  - `modules.py` contains reusable NN building blocks (attention, resblocks, positional encodings incl. chain-relative and rotary variants) and learning rate scheduler `LinearWarmupCosineDecay`.
- `src/protpardelle/data/` - Data abstractions
  - `pdb_io.py` feature extraction from PDB/mmCIF -> tensors and PDB writers (`load_feats_from_pdb`, `write_coords_to_pdb`). Respect `chain_residx_gap` logic when generating multi-chain residues.
  - `dataset.py` dataset + batching utilities (`PDBDataset`, cropping / random rotations, `make_fixed_size_1d`).
  - `motif.py` parses contig spec strings (RFdiffusion-like) into motif placement indices.
  - `atom.py` masks and conversions between backbone-only and full atom (37) representations.
- `src/protpardelle/integrations/` - External model hooks (ESMFold eval, ProteinMPNN sequence design) - loaded lazily based on flags (`num_mpnn_seqs`).
- `src/protpardelle/common/` - Static biochemical constants (`residue_constants.py`) plus lightweight protein data containers (`protein.py`: `Protein`, `Hetero`, `PDB_CHAIN_IDS`) reused across I/O, sampling, training and PDB serialization (`to_pdb`).
- `src/protpardelle/sample.py` - CLI (Typer) sampling orchestrator; builds search space combos, runs model sampling, optional ProteinMPNN sequence design, evaluation (self-consistency) and writes results under `results/` (or `PROTPARDELLE_OUTPUT_DIR`).
- `src/protpardelle/train.py` - CLI training loop with mixed precision + (optionally) `nn.DataParallel`.
- `src/protpardelle/evaluate.py` - Sequence design and self-consistency utilities.


### 2. Configuration & Execution

- Configs live under `examples/sampling/*.yaml` and `examples/training/*.yaml`. They are not Hydra multilevel packages; sampling builds Cartesian products over `search_space` lists manually (see `sample.py`). Maintain parameter names exactly (e.g. `step_scales`, `schurns`, `crop_cond_starts`).
- Environment variables (see README) override auto-detected paths: `PROTPARDELLE_MODEL_PARAMS`, `PROTEINMPNN_WEIGHTS`, `ESMFOLD_PATH`, `PROTPARDELLE_OUTPUT_DIR`, `FOLDSEEK_BIN`.
- Model checkpoints + configs stored under `model_params/` (subfolders `configs/` + `weights/`). Loading expects a config name matching weight stems (e.g. `cc58_epoch416.pth` with `configs/cc58.yaml`).

### 3. Sampling Workflow (critical path)

1. Read motif / input structure (if any) via `load_feats_from_pdb` -> features dict.
2. Build search space (product of user lists) -> each combination calls model inference.
3. Diffusion loop uses noise schedule in `diffusion.py`; `schurn` injects extra stochasticity (scaled internally by step count).
4. Crop / motif / hotspot conditioning implemented via coordinate masking (see `apply_crop_cond_strategy` in `models.py`). Sidechain tip conditioning uses curated atom lists in `residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS`.
5. Optional sequence design: ProteinMPNN (`integrations/protein_mpnn.py`) or MiniMPNN head (sequence diffusion / self-conditioning modes `seqdes`).
6. Write PDBs: for backbone-only samples first convert with `bb_coords_to_atom37_coords` if needed; chain IDs resolved either by provided mapping or sequential from `'A'` upward.

Key invariants: ensure `residue_index` starts at 1 post normalization; apply `add_chain_gap` only once; keep shape conventions `(B, N, A, 3)` for coords, `(B, N)` for indices/masks.

### 4. Training Workflow

- Entry: `python -m protpardelle.train <model_name> <output_dir>` (often submitted via `scripts/train.sbatch`).
- Datasets assembled from YAML config fields referencing prepared AI-CATH / interface data (see README dataset section). Sampling noise schedule functions: `uniform`, `lognormal`, `mpnn`, `constant` (must match those in `diffusion.noise_schedule`).
- Losses: masked MSE for coordinates (`masked_mse_loss`) + optionally cross-entropy for sequence tokens (`masked_cross_entropy_loss`). When adding a new loss term, wire it in inside the training loop where `total_loss` is accumulated (search for existing accumulation pattern). Maintain masking semantics (divide by sum(mask) with clamp >=1e-6).

### 5. Data & Tensor Conventions

- Atom ordering fixed by `residue_constants.atom_order`; backbone indices stored in `bb_idxs` in `CoordinateDenoiser`.
- For all-atom 37 representation; for backbone-only models, sometimes an extra dummy atom is added if convolution branch requires even channel counts (`if self.use_conv and n_atoms == 37: n_atoms += 1`). Preserve this logic when altering architecture.
- Chain indices are integer (0-based) while chain IDs in PDB are letters. The mapping is reversible via `chain_id_mapping` passed into writers.
- Masking: `atom_mask` shape `(N, 37)`; when absent, reconstructed from aatype (`atom37_mask_from_aatype`). Sequence mask denotes which residues are designable / present.

### 6. Extending the Model

- To add a new conditioning modality (e.g., distance map):
  1. Extend feature construction (likely in `sample.py` and training dataset assembly) producing a tensor aligned with `(B, N, A, 3)` or `(B, N, F)`.
  2. Modify `CoordinateDenoiser` input channel calculation (`nc_in`) ensuring ordering appended after existing xyz/selfcond/hotspot/ssadj channels.
  3. Update forward pass to concat new conditioning before projection into transformer / U-ViT blocks.
- To add a new noise schedule: implement in `diffusion.noise_schedule` with a new literal name; update any config enumerations and validation.

### 7. Common Pitfalls

- Forgetting to normalize `residue_index` (must start at 1) causes relative positional encoding mismatches.
- Applying `add_chain_gap` twice inflates residue indices and breaks relative chain encodings (search for single call in `load_feats_from_pdb`).
- Incorrect atom name introduces silent skip during PDB load (`if np.sum(mask) < 0.5: continue`). Provide canonical atoms or they’re dropped.
- When creating PDBs for backbone-only generation: ensure 4 atoms ordering (`N, CA, C, O`) else `bb_coords_to_pdb_str` may mis-assign residue boundaries.

### 8. Testing & Minimal Verification

- Existing automated tests sparse (`tests/test_motifbench.py` only) – integrate new logic with small unit tests using synthetic coords and motif indices (follow tensor shapes above).
- For quick smoke test after changes: load a small model config (e.g. `cc58`), run a single sample with `--num-samples 1 --num-mpnn-seqs 0` and confirm PDB output + no NaNs.

### 9. Style & Dependencies

- Python 3.12 / `uv` for dependency pinning; avoid adding large new deps unless essential. Use existing utilities in `utils.py` for seeding, device selection, path normalization.
- Logging via stdlib `logging`; prefer `logger.info` / `logger.warning` over print.

### 10. When Unsure

Reference: `models.py` for forward & sampling loops; `sample.py` for end-to-end pipeline; `pdb_io.py` for I/O conventions. Mirror existing patterns before refactoring. Ask for clarification if introducing API-breaking changes to CLI flags or config keys.

### 11. ASCII only for generated code

- All generated code (source, diffs, patches) must use plain ASCII characters.
- Allowed: standard punctuation, quotes ' " , parentheses, brackets, braces, hash for comments, backticks for Markdown.
- Disallowed in code: Unicode arrows, en/em dashes, smart quotes, multiplication dot, fancy minus, ellipsis character, box drawing characters, mathematical symbols.
- Replace stylistic arrows with `->`, long dashes with `-`, ellipsis with `...`.
- Documentation sections may use Markdown backticks but avoid decorative Unicode for consistency.
