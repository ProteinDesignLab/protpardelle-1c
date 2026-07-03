#!/bin/bash
# Sample from an overfit cc89 model at a specific epoch.
#
# Usage: bash scripts/sample_8fns_overfit.sh <epoch> [run_name]
#   epoch:    Epoch number to sample from
#   run_name: Name of the training run directory (found under cc89-overfit/cc89-overfit/)
#
# Example:
#   bash scripts/sample_8fns_overfit.sh 500 cc89-overfit-8fns

set -euo pipefail

EPOCH=${1:?Usage: $0 <epoch> [run_name]}
RUN_NAME=${2:-""}

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
cd /home/users/tianyulu/code/protpardelle-1c

TRAIN_BASE="/scratch/users/tianyulu/rhaven/cc89_overfit/cc89-overfit"
MODEL_PARAMS="$(pwd)/model_params"
MOTIF_PDB="$(pwd)/data/8fns_overfit_augmented_ingraham_cath_bugfree/motifs/8fns_with_metals.pdb"

# Find the training run directory
if [ -z "$RUN_NAME" ]; then
    # Auto-detect: find the most recent non-debug run
    RUN_NAME=$(ls -t "$TRAIN_BASE" | grep -v debug | grep -v wandb | head -1)
fi

TRAIN_DIR="$TRAIN_BASE/$RUN_NAME"
echo "Training directory: $TRAIN_DIR"
echo "Epoch: $EPOCH"

# Check checkpoint exists
CKPT="$TRAIN_DIR/checkpoints/epoch${EPOCH}_model.pth"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    ls "$TRAIN_DIR/checkpoints/" 2>/dev/null | grep model.pth | head -10
    exit 1
fi

# Symlink config and weights into model_params
mkdir -p "$MODEL_PARAMS/configs" "$MODEL_PARAMS/weights"
ln -sf "$TRAIN_DIR/config.yml" "$MODEL_PARAMS/configs/cc89-overfit.yaml"
ln -sf "$CKPT" "$MODEL_PARAMS/weights/cc89-overfit_epoch${EPOCH}.pth"

echo "Config: $MODEL_PARAMS/configs/cc89-overfit.yaml"
echo "Weights: $MODEL_PARAMS/weights/cc89-overfit_epoch${EPOCH}.pth"

# Update the overfit_test.yaml epoch
SAMPLING_YAML="examples/sampling/overfit_test.yaml"
sed -i "s/EPOCH_NUM/${EPOCH}/" "$SAMPLING_YAML"

echo ""
echo "Sampling with partial diffusion at rewind steps [90, 180, 250, 480]..."
python -u -m protpardelle.sample "$SAMPLING_YAML" \
    --motif-pdb "$MOTIF_PDB" \
    --num-samples 4 \
    --num-mpnn-seqs 0 \
    --batch-size 4

# Restore EPOCH_NUM placeholder
sed -i "s/${EPOCH}/EPOCH_NUM/" "$SAMPLING_YAML"

echo ""
echo "Sampling complete! Results in: results/overfit_test/"
