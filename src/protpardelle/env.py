"""Environment variables and path definitions.

Author: Zhaoyang Li
"""

import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from protpardelle.utils import norm_path


def _detect_project_root_dir() -> Path:
    """Determine the project root directory with the following priority:

    1) Explicit environment variable PROJECT_ROOT_DIR
    2) Search upward from current file for directory containing pyproject.toml
    3) git rev-parse --show-toplevel
    4) Fallback: parent directory of the package src directory
    """

    project_root_dir_env = os.getenv("PROJECT_ROOT_DIR")
    if project_root_dir_env is not None:
        return norm_path(project_root_dir_env)

    here = norm_path(__file__)
    for parent_dir in (here, *here.parents):
        if (parent_dir / "pyproject.toml").is_file():
            return parent_dir

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            return norm_path(out)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return here.parents[2]


PROJECT_ROOT_DIR = _detect_project_root_dir()
PACKAGE_ROOT_DIR = resources.files(__package__)

_default_protpardelle_model_params = PROJECT_ROOT_DIR / "model_params"
_default_esmfold_path = _default_protpardelle_model_params / "ESMFold"
_default_protein_mpnn_weights = (
    _default_protpardelle_model_params / "ProteinMPNN/vanilla_model_weights"
)
_default_ligand_mpnn_weights = _default_protpardelle_model_params / "LigandMPNN"
_default_protpardelle_output_dir = PROJECT_ROOT_DIR / "results"

PROTPARDELLE_MODEL_PARAMS = norm_path(
    os.getenv("PROTPARDELLE_MODEL_PARAMS", str(_default_protpardelle_model_params))
)
ESMFOLD_PATH = norm_path(os.getenv("ESMFOLD_PATH", str(_default_esmfold_path)))
PROTEINMPNN_WEIGHTS = norm_path(
    os.getenv("PROTEINMPNN_WEIGHTS", str(_default_protein_mpnn_weights))
)
LIGANDMPNN_WEIGHTS = norm_path(
    os.getenv("LIGANDMPNN_WEIGHTS", str(_default_ligand_mpnn_weights))
)
PROTPARDELLE_OUTPUT_DIR = norm_path(
    os.getenv("PROTPARDELLE_OUTPUT_DIR", str(_default_protpardelle_output_dir))
)

if (
    _default_foldseek_bin_env := os.getenv("FOLDSEEK_BIN", shutil.which("foldseek"))
) is None:
    raise ValueError("Foldseek executable not found")
FOLDSEEK_BIN = norm_path(_default_foldseek_bin_env)

__all__ = [
    "PROJECT_ROOT_DIR",
    "PACKAGE_ROOT_DIR",
    "PROTPARDELLE_MODEL_PARAMS",
    "ESMFOLD_PATH",
    "PROTEINMPNN_WEIGHTS",
    "LIGANDMPNN_WEIGHTS",
    "PROTPARDELLE_OUTPUT_DIR",
    "FOLDSEEK_BIN",
]
