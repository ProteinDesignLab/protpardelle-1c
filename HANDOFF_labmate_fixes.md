# Protpardelle-1c — Fixes & Setup Handoff (labmate onboarding)

**Audience:** anyone (or their Claude session) hitting crashes / broken paths / 0-success
binder runs on the `pd_motif_and_latent` branch.
**Author:** Tianyu Lu · **Date:** 2026-07-19 · **Branch:** `pd_motif_and_latent`

> **If you are a Claude session reading this:** this file is a task list. Work top to
> bottom. Section 1 is code fixes that are **already committed** — the user just needs to
> pull. Sections 2–3 are **per-user environment setup** you should apply to their configs /
> shell. Section 4 is a design-quality diagnosis (why a PD-L1 run got 0/32). Verify each item
> against the live code before telling the user it's done — line numbers may drift.

---

## 0. First: get the fixes

```bash
cd <your protpardelle-1c checkout>
git fetch origin
git log --oneline -3 origin/pd_motif_and_latent
# You want these two commits present:
#   8d87034  Allow overriding Caliby venv_python via $CALIBY_VENV_PYTHON
#   8029a74  Fix unconditional sampling crash, HF-hub ESMFOLD_PATH, and Caliby GPU contention
git merge --ff-only origin/pd_motif_and_latent   # or rebase your work onto it
```

If those two commits are already in your history, **all of Section 1 is done** — skip to
Section 2.

---

## 1. Code bugs fixed (commits `8029a74`, `8d87034`)

### 1.1 Unconditional sampling (`motifs: [null]`) crashed with `FileNotFoundError`
- **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'motifs/<dir>/000_unconditional'`,
  dies immediately, before any sampling.
- **Cause:** the motif-provenance `shutil.copy(motif_fp, per_motif_save_dir)` added in commit
  `8e3d389` ran in the plain single-motif branch even for the unconditional **placeholder** path
  `motif_dir/"NNN_unconditional"`, which never points at a real file.
- **Fix location:** `src/protpardelle/sample.py`, in `sample()`, the `else` branch of the
  motif-copy block (~line **1923**, look for the comment `# Unconditional runs (motifs: [null])`).
  The copy + `_write_conditioning_atoms_pdb` are now guarded by `if Path(motif_fp).exists():`.
- **Verify:** `python -m protpardelle.sample examples/sampling/uncond_test.yaml --num-samples 1 --num-mpnn-seqs 0`
  should run to completion (was a hard crash before).

### 1.2 Bare Hugging Face Hub `ESMFOLD_PATH` was mangled by `norm_path()`
- **Symptom:** setting `ESMFOLD_PATH=facebook/esmfold_v1` produces a bogus local path
  (`<cwd>/facebook/esmfold_v1`) and an `OSError` from transformers.
- **Cause:** `norm_path()` unconditionally calls `Path(...).resolve()`, which rewrites a bare
  `org/repo` Hub id relative to the current directory.
- **Fix location:** `src/protpardelle/env.py`
  - new helper `_looks_like_hf_hub_id()` (~line **16**)
  - `ESMFOLD_PATH` assignment now passes Hub ids through verbatim (~line **80–87**).
- **Effect:** you may now set `ESMFOLD_PATH=facebook/esmfold_v1` directly (no pre-download
  needed), **or** point it at a local snapshot dir — both work.

### 1.3 Caliby subprocess: `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- **Symptom:** backbone/ensemble generation finishes, then the Caliby sequence-design subprocess
  dies at model load / `torch.load` device transfer.
- **Cause:** on GPU nodes in **EXCLUSIVE_PROCESS** compute mode, the subprocess can't open a
  second CUDA context on the same GPU the main sampling process already holds one on. The
  subprocess inherited `CUDA_VISIBLE_DEVICES` unchanged.
- **Fix location:** `src/protpardelle/sample.py`
  - `_resolve_caliby_cuda_device()` (~line **183**) — reads `caliby_ensemble.cuda_device`, then
    `$CALIBY_CUDA_VISIBLE_DEVICES`.
  - `_call_caliby_subprocess_batched()` sets `env["CUDA_VISIBLE_DEVICES"]` when a device is given,
    and calls `torch.cuda.empty_cache()` before launch (~lines **254–263**).
  - threaded into both call sites (`run_inference_guidance`, `run_caliby_ensemble_design`).
- **How to use it → see Section 2.2.**

### 1.4 Caliby `venv_python` pointed at a private, non-group-readable path
- **Symptom:** `venv_python: /scratch/users/tianyulu/envs/caliby/bin/python` in the shared
  configs is a symlink into `/home/users/tianyulu/.local/...` ($HOME, not group-readable) — so
  caliby sequence design is broken for everyone except the config author.
- **Fix location:** `src/protpardelle/sample.py`, `_resolve_caliby_venv_python()` (~line **171**):
  resolves `$CALIBY_VENV_PYTHON` first, falling back to the config value.
- **How to use it → see Section 2.1.**

---

## 2. Per-user environment setup (do this once)

These are **not** code changes — they are env vars / config values each user must set for their
own account. Put them in your sbatch script or shell before running `python -m protpardelle.sample`.

```bash
# Model weights & configs (group-readable, unchanged):
export PROTPARDELLE_MODEL_PARAMS=/oak/stanford/groups/possu/tianyu/models/protpardelle-1c

# Foldseek — use the group-shared binary (no download needed):
export FOLDSEEK_BIN=/home/groups/possu/foldseek/bin/foldseek

# ESMFold — either the bare Hub id (now works, see 1.2) or a local snapshot:
export ESMFOLD_PATH=facebook/esmfold_v1
```

### 2.1 Caliby venv (`$CALIBY_VENV_PYTHON`)
Caliby is public: <https://github.com/ProteinDesignLab/caliby> (uv venv + `uv pip install -e .`).
Build your own once, then point at it — **do not edit the shared config**:

```bash
export CALIBY_VENV_PYTHON=/scratch/users/<you>/envs/caliby/bin/python
```
The Caliby checkpoints (`model_params_dir: /oak/stanford/groups/possu/tianyu/models/caliby`) are
already group-readable — leave that config value alone.

### 2.2 Caliby GPU (avoid the device-busy crash)
If your node's GPUs are in EXCLUSIVE_PROCESS mode (or you just want isolation), request **two**
GPUs and route Caliby to the second one:

```bash
#SBATCH -G 2
export CALIBY_CUDA_VISIBLE_DEVICES=1   # main process uses GPU 0, Caliby subprocess uses GPU 1
```
Alternatively set `caliby_ensemble.cuda_device: 1` in your sampling sub-config. On Default
compute-mode nodes you can skip this (sharing one GPU works); the fix also frees the main
process's GPU cache before launching Caliby to give a shared GPU headroom.

### 2.3 Note: configs without a `caliby_ensemble` block silently skip Caliby
If Caliby "didn't run" and there was no error, check that your sampling sub-config actually has a
`caliby_ensemble:` block with `enabled: true` (e.g.
`sampling_sidechain_conditional_allatom_s1_caliby.yaml`). The plain `..._s1` variants have no such
block and skip Caliby by design.

---

## 3. Quick smoke test (confirm your pipeline is healthy end-to-end)

```bash
python -m protpardelle.sample examples/sampling/04_bindcraft.yaml \
    --motif-dir examples/motifs/bindcraft \
    --num-samples 8 --num-mpnn-seqs 8
```
This is the **validated** binder-generation config (see Section 4). Everything it needs is in the
repo + on oak. Expect some samples with `ca_scaffold_scrmsd < 2Å` (self-consistency passes).

---

## 4. Design-quality diagnosis — "0/32 self-consistency on PD-L1"

If you conditioned a **binder** run on hotspots and got ~0 self-consistent designs *despite tight
motif placement* (~0.06Å), the geometry being satisfiable does **not** mean the setup is right.
Check these three levers against the known-good reference `examples/sampling/04_bindcraft.yaml`
(target `1_PD-L1`, motif `examples/motifs/bindcraft/1_PD-L1.pdb`):

| Lever | Common mistake | Validated PD-L1 setting |
|---|---|---|
| **Model** | `cc145` (that's the *motif-scaffolding* / genie3 model) | **`cc83` epoch `2616`** with `sampling_sidechain_conditional` (the *binder* model) |
| **Hotspots** | arbitrary surface residues in native numbering (e.g. A56/A115/A123) | **A37, A39, A49, A98** = Ile/Tyr/Gln/Met, the PD-1/PD-L1 interface patch (motif renumbered A1-115) |
| **Contig / length** | `A17-132;/;50-120` | `A1-115;/;65-155` |

Notes:
- **Residue numbering matters.** The validated motif is renumbered A1-115; a raw/repacked target
  may start ~res 17 (~+16 offset), so native-numbered hotspots won't map onto the validated set.
  Always confirm your hotspot residue numbers refer to the *interface* residues in *your* motif
  file's numbering.
- **Checkpoint recency (if you must use cc145):** use epoch **199** or **273swa8**, never 170 —
  all the maintained cc145 benchmark configs (`examples/sampling/genie3_benchmark_cc145_e199.yaml`,
  `..._e273swa8.yaml`) do. `273swa8` = weights averaged over the last 8 epochs (SWA).
- **Sampling hyperparameters are usually NOT the lever.** Per the README, start with ODE
  (`step_scale=1.0, schurn=0`) when evaluating a *new* model, and note `schurn=200` is poor for
  multi-chain all-atom models — but the maintained benchmarks still run `ss=1.2/schurn=200`
  successfully, so if you're getting 0 success, fix model + hotspots first, not schurn.

---

## 5. Where to look in the code (quick index)

| Concern | File · symbol |
|---|---|
| Sampling entry point | `src/protpardelle/sample.py` · `sample()`, `generate()`, `draw_samples()` |
| Unconditional-copy guard | `src/protpardelle/sample.py` ~L1923 (`# Unconditional runs`) |
| Caliby venv/gpu resolvers | `src/protpardelle/sample.py` · `_resolve_caliby_venv_python`, `_resolve_caliby_cuda_device` |
| Caliby subprocess launcher | `src/protpardelle/sample.py` · `_call_caliby_subprocess_batched` |
| Env vars & path normalization | `src/protpardelle/env.py` · `_looks_like_hf_hub_id`, `ESMFOLD_PATH`, `FOLDSEEK_BIN` |
| Caliby subprocess helper (runs in caliby venv) | `scripts/run_caliby_ensemble.py` |
| Broader project map | `CLAUDE.md` (repo root) |
