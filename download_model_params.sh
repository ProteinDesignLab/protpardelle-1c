#!/bin/bash

set -euo pipefail

# Parse command line arguments
DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dev]"
            echo ""
            echo "Options:"
            echo "  --dev    Create empty directories only, skip downloads"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Check required commands only if not in dev mode
if [[ "$DEV_MODE" == "false" ]]; then
    for cmd in wget curl aria2c hf; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: $cmd is required but not installed." >&2
            exit 1
        fi
    done
fi

mkdir -p model_params

# Download Protpadelle model params
if [[ "$DEV_MODE" == "true" ]]; then
    mkdir -p model_params/configs
    mkdir -p model_params/weights
else
    echo "Downloading Protpadelle model parameters..."
    aria2c -x16 -s16 -o protpardelle-1c.tar.gz "https://zenodo.org/records/16817230/files/protpardelle-1c.tar.gz?download=1"
    tar -xzvf protpardelle-1c.tar.gz --strip-components=1
    rm protpardelle-1c.tar.gz
    echo "Protpadelle model parameters downloaded."
fi

# Download ESMFold model
if [[ "$DEV_MODE" == "true" ]]; then
    mkdir -p model_params/ESMFold
else
    echo "Downloading ESMFold model..."
    mkdir -p model_params/ESMFold
    hf download facebook/esmfold_v1 --local-dir model_params/ESMFold
    echo "ESMFold model downloaded."
fi

# Download ProteinMPNN weights
if [[ "$DEV_MODE" == "true" ]]; then
    mkdir -p model_params/ProteinMPNN/vanilla_model_weights
else
    echo "Downloading ProteinMPNN weights..."
    mkdir -p model_params/ProteinMPNN
    tmp="$(mktemp -d)"
    repo_url="https://github.com/dauparas/ProteinMPNN.git"
    branch="main"
    folder="vanilla_model_weights"

    git_ver="$(git --version | awk '{print $3}')"
    IFS=. read -r M m p <<<"$git_ver"
    : "${m:=0}"
    : "${p:=0}"
    # Check Git version >= 2.25.0
    if ((M > 2 || (M == 2 && (m > 25 || (m == 25 && p >= 0))))); then
        git clone --depth=1 --filter=tree:0 --sparse "$repo_url" "$tmp"
        git -C "$tmp" sparse-checkout set "$folder"
        git -C "$tmp" checkout "$branch"
    else
        git clone --depth=1 --single-branch --branch "$branch" "$repo_url" "$tmp" \
            || git clone --depth=1 "$repo_url" "$tmp"
    fi

    mv "$tmp/$folder" model_params/ProteinMPNN/
    rm -rf "$tmp"
    echo "ProteinMPNN weights downloaded."
fi

# Download LigandMPNN weights
if [[ "$DEV_MODE" == "true" ]]; then
    mkdir -p model_params/LigandMPNN
else
    echo "Downloading LigandMPNN weights..."
    mkdir -p model_params/LigandMPNN
    curl -fsSL https://raw.githubusercontent.com/dauparas/LigandMPNN/main/get_model_params.sh | bash -s -- ./model_params/LigandMPNN
    echo "LigandMPNN weights downloaded."
fi
