# Protpardelle-1c

The original Protpardelle is detailed in our paper [An all-atom protein generative model](https://www.pnas.org/doi/10.1073/pnas.2311500121). Protpardelle-1c provides new models which achieve more robust motif scaffolding results benchmarked with [MotifBench](https://github.com/blt2114/MotifBench) and [La-Proteina](https://research.nvidia.com/labs/genair/la-proteina/). The code includes examples of unconditional sampling, partial diffusion, motif scaffolding, binder generation, and model training. For more details on the benchmarking results, new multichain support, architectural and dataset changes, please see our [preprint](https://www.biorxiv.org/content/10.1101/2025.08.18.670959v2).

# Installation

## Prerequisites

To run the scripts in this repository, we recommend using `conda` for environment management and `uv` for python dependency management. If you don't have `conda` installed yet, you can follow the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install); `uv` installation will be automatically handled within `setup.sh`.

This repository is tested on Linux with `gcc>=12.4` and `cuda>=12.4`.

## Install dependencies

Next, you can clone the repository and install its dependencies by running:

```bash
# Clone the repository
git clone https://github.com/ProteinDesignLab/protpardelle-1c.git
cd protpardelle-1c

# Create and activate a conda environment
conda create -n protpardelle python=3.12 --yes
conda activate protpardelle

# Install dependencies using `setup.sh`
# Note 1: 
# Note 2: Use a recent version of gcc. If working on a cluster, this might look like "module load gcc/12.4.0"
bash setup.sh
```

If working on a cluster with limited home storage, set `CONDA_PKGS_DIRS` and `UV_CACHE_DIR` to a directory with higher storage quotas.

The `uv.lock` file is a example environment with exact versions of all installed packages. If you prefer to only use `uv`, the dependencies can be installed by running:

```bash
ENV_DIR=envs  # or any other directory of your choice
mkdir -p $ENV_DIR
uv venv $ENV_DIR/protpardelle -p python3.10
source $ENV_DIR/protpardelle/bin/activate

git clone https://github.com/ProteinDesignLab/protpardelle-1c.git
cd protpardelle-1c

uv pip sync uv_indexes.txt uv.lock  --index-strategy=unsafe-best-match
uv pip install git+https://github.com/sokrypton/openfold.git@4fbff9bc73d867be19594fe4d135875566162de3 --no-build-isolation
uv pip install -e .
```

## Download model weights and configs

Download the pre-trained model weights and corresponding configs from [Zenodo](https://zenodo.org/records/16817230). To run sampling with an all-atom model, download the original [ProteinMPNN weights](https://github.com/dauparas/ProteinMPNN/tree/main/vanilla_model_weights). To run evaluation, download the [ESMFold weights](https://huggingface.co/facebook/esmfold_v1) from Hugging Face.

We use [aria2](https://github.com/aria2/aria2) and [`huggingface-hub[cli]`](https://pypi.org/project/huggingface-hub/) in our download script. All downloads are automatically handled by running

```bash
bash download_model_params.sh
```

It takes some time to download all the files, and you should see the following directories created:

- `<project_root>/model_params/`
- `<project_root>/model_params/ESMFold/`
- `<project_root>/model_params/ProteinMPNN/vanilla_model_weights/`
- `<project_root>/model_params/configs/`
- `<project_root>/model_params/weights/`

Then install [Foldseek](https://github.com/steineggerlab/foldseek) in your `PATH` following their instructions.

Outputs will be saved in `<project_root>/results` by default.

## Set environment variables

> If you keep the default layout above, you can skip this section.

If the `foldseek` binary is already on your `PATH`, you do not need to set `FOLDSEEK_BIN`; otherwise set it explicitly (see below). To override file locations, set the following variables to absolute paths:

```bash
# (Optional) only if auto-detection of project root fails:
export PROJECT_ROOT_DIR=/abs/path/to/your/protpardelle-repo

# Model weights & configs (directory)
export PROTPARDELLE_MODEL_PARAMS=/abs/path/to/model_params

# ESMFold weights (directory)
export ESMFOLD_PATH=/abs/path/to/ESMFold

# ProteinMPNN weights (directory)
export PROTEINMPNN_WEIGHTS=/abs/path/to/ProteinMPNN/vanilla_model_weights

# Default output directory (optional; defaults to <project_root>/results)
export PROTPARDELLE_OUTPUT_DIR=/abs/path/to/output_dir

# Foldseek binary path (set this if foldseek is not already on PATH)
export FOLDSEEK_BIN=/abs/path/to/foldseek/bin
```

## Available Models

| Name | Monomers | Multichain | Model Type | Positional Encoding | Notes |
|------|----------|------------|------------|-------------------|-------|
| **bb81_epoch450** | 1 | 0 | Backbone | Rotary | Unconditional model trained on AI-CATH |
| **bbmd_epoch500** | 1 | 0 | Backbone | Rotary | Unconditional model trained on MD-CATH |
| **cc58_epoch416** | 1 | 0 | Backbone | Rotary | MotifBench benchmark model |
| **cc58_epochX** | 1 | 0 | Backbone | Rotary | Additional checkpoints of cc58 |
| **cc58-minimpnn** | 1 | 0 | Sequence design | - | Trained on cc58_epoch595 1-step x0 predicted structures |
| **cc78_epoch1431** | 0 | 1 | Backbone | Relative + Relchain | Experimental: residue index are tied across chains, favors homodimers |
| **cc83_epoch2616** | 0.5 | 0.5 | Backbone | Relative | Bindcraft benchmark model |
| **cc89_epoch415** | 1 | 0 | Allatom Sequence Mask | Rotary | Sequence must be provided at all sampling steps |
| **cc91_epoch383** | 1 | 0 | Allatom No Mask | Relative | Allatom model trained on AI-CATH |
| **cc91_tip_epoch480** | 1 | 0 | Allatom No Mask | Relative | cc91 finetuned on sidechain tip atom conditioning task |
| **cc94_epoch3100** | 0.5 | 0.5 | Allatom No Mask | Relative | cc91 finetuned on multichain data but no hotspot |
| **cc95_epoch3490** | 0.5 | 0.5 | Backbone | Relative + Relchain | cc83 finetuned with heavier hotspot dropout |

# Output

The outputs are saved in a nested folder structure where the levels are organized as such:

```

PROTPARDELLE_OUTPUT_DIR
└── sampling-experiment-name
    └── model-epoch-sampling_config-stepscale-schurn-ccstart-dx-dy-dz-rewind
        └── 7eow_CDR3_atom_rot_128

```

For example, the demo `02_motif_scaffolding` with default sampling settings will generate the following folder structure under `PROTPARDELLE_OUTPUT_DIR`:

```

02_motif_scaffolding
└── cc58-epoch416-sampling_sidechain_conditional-ss1.2-schurn200-ccstart0.0-dx0.0-dy0.0-dz0.0-rewindNone
    └── 7eow_CDR3_atom_rot_128
        └── scaffold_info.csv
            7eow_CDR3_atom_rot_128_0.pdb
            ...
└── cc95-epoch3490-sampling_sidechain_conditional-ss1.2-schurn200-ccstart0.0-dx0.0-dy0.0-dz0.0-rewindNone
    └── 7eow_CDR3_atom_rot_128
        └── scaffold_info.csv
            7eow_CDR3_atom_rot_128_0.pdb
            ...
└── cc94-epoch3100-sampling_sidechain_conditional_allatom-ss1.2-schurn200-ccstart0.0-dx0.0-dy0.0-dz0.0-rewindNone
    └── 7eow_CDR3_atom_rot_128
        └── scaffold_info.csv
            7eow_CDR3_atom_rot_128_0.pdb
            ...

```

This folder organization, in particular `scaffold_info.csv`, follow MotifBench input specifications.

# Sampling Examples

See `examples/sampling` for demos:

```bash
python3 scripts/eval_samples.py ./examples/sampling/00_unconditional.yaml --n-samples 8 --num-mpnn-seqs 0 --debug
python3 scripts/eval_samples.py ./examples/sampling/01_partial_diffusion.yaml --motif-dir ./examples/motifs/nanobody --n-samples 8 --num-mpnn-seqs 0 --debug
python3 scripts/eval_samples.py ./examples/sampling/02_motif_scaffolding.yaml --motif-dir ./examples/motifs/nanobody --n-samples 8 --num-mpnn-seqs 0 --debug

# MotifBench
python3 scripts/eval_samples.py ./examples/sampling/03_motifbench.yaml --motif-dir ./examples/motifs/motif_bench --n-samples 100 --num-mpnn-seqs 8 --debug

# BindCraft
python3 scripts/eval_samples.py ./examples/sampling/04_bindcraft.yaml --motif-dir ./examples/motifs/bindcraft/ --n-samples 100 --num-mpnn-seqs 2 --debug

python3 scripts/eval_samples.py ./examples/sampling/05_multichain.yaml --motif-dir ./examples/motifs/nanobody/ --n-samples 8 --num-mpnn-seqs 0 --debug
python3 scripts/eval_samples.py ./examples/sampling/06_fastrelax.yaml --motif-dir ./examples/motifs/nanobody/ --n-samples 8 --num-mpnn-seqs 0 --debug

# La-Proteina / RFdiffusion benchmark
python3 scripts/eval_samples.py ./examples/sampling/07_rfdiffusion.yaml --motif-dir ./examples/motifs/rfdiffusion/ --n-samples 200 --num-mpnn-seqs 1 --debug
python3 scripts/eval_samples.py ./examples/sampling/08_rfdiffusion_allatom.yaml --motif-dir ./examples/motifs/rfdiffusion/ --n-samples 200 --num-mpnn-seqs 1 --debug
python3 scripts/eval_samples.py ./examples/sampling/09_structure_prediction.yaml --motif-dir ./examples/motifs/nanobody/ --n-samples 8 --num-mpnn-seqs 0 --debug
```

## Sampling configs

Multiple entries can be specified per setting in `search_space` and are combined with `itertools.product()`. The entries `motifs`, `motif_contigs`, `total_lengths`, `hotspots`, and `ssadj`, must be equal in number, i.e. two entries under `motifs` must be matched with two `motif_contigs`, etc.

### `search_space.models`

Model weights to use for sampling formatted as `[model_name, epoch, sampling_config]`. The default backbone-only model is `cc58` epoch `416`. The default allatom model is `cc94` epoch `3100`. The default multichain model is `cc95` epoch `3490`.

### `search_space.step_scales`

List of floats where the model's score is scaled by this amount, where higher step scales correspond to lower temperature sampling and lower step scales correspond to higher temperature sampling. Recommended setting is `1.2`.

### `search_space.schurns`

List of floats where higher magnitudes inject more stochasticity during denoising and `0` is noise-free. The combination of step scale `1.0` and schurn `0` corresponds to ODE sampling. We recommend to always start with ODE sampling as this is exactly the denoising path the model has learned, without any _post hoc_ corrections. The same `schurn` may not generalize across different models; we have observed poor sample quality with `schurn=200` for the all-atom models while `schurn=200` is okay for backbone-only models.

### `search_space.crop_cond_starts`

List of floats in `[0.0, 1.0]` indicating the fraction of total denoising steps to start applying crop-conditional guidance.

### `search_space.translations`

Translate the input motif by `[x, y, z]` Ångstroms.

### `motifs`

Stem of the `.pdb` file containing the motif to scaffold. The file should be located under the `{MOTIF_DIR}` given to `eval_samples.sbatch`. Set as `null` for unconditional sampling and set to the path to the full structure for partial diffusion.

### `motif_contigs`

For partial diffusion, write `partial_diffusion`. For motif scaffolding, the contig follows similar syntax as RFdiffusion but we use `;/;` to denote a chain break.

#### Example contigs

- `0-100;A1-21;0-100`: sample 0 to 100 scaffold residues, followed by motif PDB's chain A residues 1 to 21, followed by 0 to 100 scaffold residues.
- `A1-128;/;120-120`: condition on chain A residues 1 to 128, generate another chain with exactly 120 residues.
- `A1-79;/;B1-141;/;C1-33;/;70-150`: condition on chains A, B, and C, generate another chain with 70 to 150 residues.

### `total_lengths`

List of list of integers indicating the number of residues per chain, where each sublist is the length range for one chain. For binder generation, the total lengths for the target chain(s) should match their number of residues.

### `hotspots`

For multichain / binder generation, a comma-delimited string with format `{chain_id}{residue_index}`.

### `ssadj`

For fold-conditioning, the stems of the output files from running the `make_secstruc_adj.py`, script from RFdiffusion. Per-residue secondary structure labels and per residue pair block adjacency contact info are encoded as conditioning inputs. Fold-conditional model weights will be released at a later date.

# Training

First change these lines in `scripts/train.sh`:

```bash
#SBATCH -p your-partition-name
PROTPARDELLE_EXPERIMENT_DIR=/your/path/to/output/dir
--wandb_id your-wandb-id
```

See `examples/training` for the exact model architecture and training hyperparameters used in Protpardelle-1c models. Look for `/your/path/to` which need to be changed to paths on your current system. The training script can be called with

```bash
source scripts/train.sbatch {EXPERIMENT_NAME}

# Check if everything works
source scripts/train.sh cc58 --debug

# Then submit job
sbatch -J cc58 scripts/train.sh cc58
```

A copy of the training config and model checkpoints will be saved under `{PROTPARDELLE_EXPERIMENT_DIR}`

## Datasets

- [AI-CATH](https://zenodo.org/records/15881564): The CATH dataset described in the original Protpardelle paper but augmented with 32 ProteinMPNN sequences per structure. Structures are predicted by ESMFold. Models are trained on only the designable subset (337,936 / 704,448).
- Boltz Interfaces: PDB chain pairs curated following Boltz, total 1,593,738 chain pairs.

## Samples

- MotifBench samples and results are at [Zenodo](https://zenodo.org/records/16651614). These were produced with the `03_motifbench` config.
- RFdiffusion/La-Proteina motif scaffolding samples and results are at [Zenodo](https://zenodo.org/records/16887802). These were produced with the `07_rfdiffusion` and `08_rfdiffusion_allatom` configs.

# Acknowledgements

We thank Sergey Ovchinnikov for the modified ESM and OpenFold repos which we use.

```
@article{ahdritz2024openfold,
  title={OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
  author={Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O’Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccol{\`o} and others},
  journal={Nature methods},
  volume={21},
  number={8},
  pages={1514--1524},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```
