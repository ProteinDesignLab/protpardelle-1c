#!/bin/bash

set -euo pipefail

if ! command -v aria2c >/dev/null 2>&1; then
    echo "Error: aria2c is required but not installed." >&2
    exit 1
fi

# Download Protpadelle model params
aria2c -x16 -s16 -o protpardelle-1c.tar.gz "https://zenodo.org/records/16817230/files/protpardelle-1c.tar.gz?download=1"
tar -xzvf protpardelle-1c.tar.gz --strip-components=1  # there will be a `model_params/` directory
rm protpardelle-1c.tar.gz

# Download ESMFold model
mkdir -p model_params/ESMFold
aria2c -x16 -s16 -o model_params/ESMFold/esmfold.model "https://colabfold.steineggerlab.workers.dev/esm/esmfold.model"

# Download ProteinMPNN weights
mkdir -p model_params/ProteinMPNN
tmp="$(mktemp -d)"
git clone --depth=1 --filter=tree:0 --sparse https://github.com/dauparas/ProteinMPNN.git "$tmp"
git -C "$tmp" sparse-checkout set vanilla_model_weights
git -C "$tmp" checkout main
mv "$tmp/vanilla_model_weights" model_params/ProteinMPNN/
rm -rf "$tmp"
