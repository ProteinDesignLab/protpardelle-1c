#!/bin/bash

set -euo pipefail

pip install uv

uv pip install torch # --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

# Install and patch ESM2
uv pip install git+https://github.com/facebookresearch/esm.git
patch "$(python -c 'import site,sys; sys.stdout.write(next(p for p in site.getsitepackages() if "site-packages" in p) + "/esm/inverse_folding/util.py")')" <<'EOF'
@@ -12,1 +12,1 @@
-from biotite.structure import filter_backbone
+from biotite.structure import filter_peptide_backbone as filter_backbone
EOF

# Install OpenFold
uv pip install git+https://github.com/aqlaboratory/openfold.git --no-build-isolation

# TODO: Fix
# uv pip sync uv.lock --index-strategy=unsafe-best-match
# uv pip install --no-build-isolation "openfold @ git+https://github.com/sokrypton/openfold.git@4fbff9bc73d867be19594fe4d135875566162de3"
# uv pip install -e .