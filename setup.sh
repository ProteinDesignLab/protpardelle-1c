#!/bin/bash

set -euo pipefail

# Parse command line arguments
INSTALL_FOLDSEEK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --foldseek)
            INSTALL_FOLDSEEK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--foldseek]"
            echo ""
            echo "Options:"
            echo "  --foldseek   Install Foldseek (CPU version) via conda after Python setup"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

pip install uv

# Install CUDA-compatible PyTorch and Protpardelle
uv pip install torch # --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

# Install Foldseek if requested
if [[ "$INSTALL_FOLDSEEK" == "true" ]]; then
    echo "Installing Foldseek via conda..."
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda is required for Foldseek installation but not found." >&2
        echo "Please install conda or mamba first." >&2
        exit 1
    fi
    
    # Check if foldseek is already installed
    if command -v foldseek >/dev/null 2>&1; then
        echo "Foldseek is already installed. Skipping installation."
    else
        conda install -c conda-forge -c bioconda foldseek -y
        echo "Foldseek installation completed."
    fi
fi
