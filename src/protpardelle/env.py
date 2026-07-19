"""Environment variables and path definitions.

Author: Zhaoyang Li
"""

import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from protpardelle.utils import norm_path


def _looks_like_hf_hub_id(value: str) -> bool:
    """Heuristic: is ``value`` a bare Hugging Face Hub repo id (``org/repo``)?

    A bare Hub id such as ``facebook/esmfold_v1`` must reach
    ``transformers.from_pretrained`` verbatim. Running it through
    ``norm_path()`` (which calls ``Path.resolve()``) would rewrite it relative
    to the current working directory and produce a bogus local path, so we must
    detect and pass these through untouched. Treat a string as a Hub id when it
    has exactly one ``/`` in ``org/repo`` shape and does not already exist on
    disk (a real local snapshot path always wins).
    """

    expanded = os.path.expanduser(os.path.expandvars(value))
    if os.path.exists(expanded):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", value))


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
_default_protpardelle_output_dir = PROJECT_ROOT_DIR / "results"

PROTPARDELLE_MODEL_PARAMS = norm_path(
    os.getenv("PROTPARDELLE_MODEL_PARAMS", str(_default_protpardelle_model_params))
)
_esmfold_path_env = os.getenv("ESMFOLD_PATH", str(_default_esmfold_path))
# Preserve bare Hub ids (e.g. "facebook/esmfold_v1") verbatim; only normalize
# real filesystem paths.
ESMFOLD_PATH = (
    _esmfold_path_env
    if _looks_like_hf_hub_id(_esmfold_path_env)
    else norm_path(_esmfold_path_env)
)
PROTEINMPNN_WEIGHTS = norm_path(
    os.getenv("PROTEINMPNN_WEIGHTS", str(_default_protein_mpnn_weights))
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
    "PROTPARDELLE_OUTPUT_DIR",
    "FOLDSEEK_BIN",
]
