#!/bin/bash

set -euo pipefail

pip install uv

# Install CUDA-compatible PyTorch and Protpardelle
uv pip install torch # --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
