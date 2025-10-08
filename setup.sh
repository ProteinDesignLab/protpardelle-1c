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
            echo "  --dev    Use development mode"
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

pip install uv

# Install CUDA-compatible PyTorch and Protpardelle
uv pip install torch # --index-url https://download.pytorch.org/whl/cu128

if [[ "$DEV_MODE" == "true" ]]; then
    uv pip install -e .[dev]
else
    uv pip install -e .
fi

