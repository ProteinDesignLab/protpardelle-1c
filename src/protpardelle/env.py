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


def _detect_package_root_dir() -> Path:
    """Determine the package root directory."""
    return Path(resources.files(__package__))  # type: ignore


PROJECT_ROOT_DIR = _detect_project_root_dir()
PACKAGE_ROOT_DIR = _detect_package_root_dir()


class _Env:
    """Environment variables and paths used in ProtPardelle."""

    @staticmethod
    def protpardelle_model_params() -> Path:
        """Determine the default model parameters directory."""
        _default_protpardelle_model_params = PROJECT_ROOT_DIR / "model_params"
        _protpardelle_model_params = norm_path(
            os.getenv(
                "PROTPARDELLE_MODEL_PARAMS",
                default=str(_default_protpardelle_model_params),
            )
        )
        if not _protpardelle_model_params.is_dir():
            raise NotADirectoryError(
                f"ProtPardelle model parameters directory not found: {_protpardelle_model_params}"
            )

        return _protpardelle_model_params

    @staticmethod
    def esmfold_path() -> Path:
        """Determine the default ESMFold path."""
        _default_esmfold_path = PROJECT_ROOT_DIR / "model_params" / "ESMFold"
        _esmfold_path = norm_path(
            os.getenv("ESMFOLD_PATH", default=str(_default_esmfold_path))
        )
        if not _esmfold_path.is_dir():
            raise NotADirectoryError(f"ESMFold path not found: {_esmfold_path}")

        return _esmfold_path

    @staticmethod
    def protein_mpnn_weights() -> Path:
        """Determine the default ProteinMPNN weights path."""
        _default_protein_mpnn_weights = (
            PROJECT_ROOT_DIR / "model_params" / "ProteinMPNN" / "vanilla_model_weights"
        )
        _protein_mpnn_weights = norm_path(
            os.getenv("PROTEINMPNN_WEIGHTS", default=str(_default_protein_mpnn_weights))
        )
        if not _protein_mpnn_weights.is_dir():
            raise NotADirectoryError(
                f"ProteinMPNN weights path not found: {_protein_mpnn_weights}"
            )

        return _protein_mpnn_weights

    @staticmethod
    def ligand_mpnn_weights() -> Path:
        """Determine the default LigandMPNN weights path."""
        _default_ligand_mpnn_weights = PROJECT_ROOT_DIR / "model_params" / "LigandMPNN"
        _ligand_mpnn_weights = norm_path(
            os.getenv("LIGANDMPNN_WEIGHTS", default=str(_default_ligand_mpnn_weights))
        )
        if not _ligand_mpnn_weights.is_dir():
            raise NotADirectoryError(
                f"LigandMPNN weights path not found: {_ligand_mpnn_weights}"
            )

        return _ligand_mpnn_weights

    @staticmethod
    def protpardelle_output_dir() -> Path:
        """Determine the default output directory."""
        _default_protpardelle_output_dir = PROJECT_ROOT_DIR / "results"
        _protpardelle_output_dir = norm_path(
            os.getenv(
                "PROTPARDELLE_OUTPUT_DIR", default=str(_default_protpardelle_output_dir)
            )
        )
        if not _protpardelle_output_dir.is_dir():
            # Do not raise error, create the directory if it does not exist
            _protpardelle_output_dir.mkdir(parents=True, exist_ok=True)

        return _protpardelle_output_dir

    @staticmethod
    def foldseek_bin() -> Path:
        """Determine the Foldseek binary path."""
        _default_foldseek_bin_env = os.getenv(
            "FOLDSEEK_BIN", default=shutil.which("foldseek")
        )
        if _default_foldseek_bin_env is None:
            raise ValueError("Foldseek executable not found")

        _foldseek_bin_env = norm_path(_default_foldseek_bin_env)
        if not _foldseek_bin_env.is_file():
            raise FileNotFoundError(
                f"Foldseek executable not found: {_foldseek_bin_env}"
            )

        return _foldseek_bin_env


PROTPARDELLE_MODEL_PARAMS = _Env.protpardelle_model_params()
ESMFOLD_PATH = _Env.esmfold_path()
PROTEINMPNN_WEIGHTS = _Env.protein_mpnn_weights()
LIGANDMPNN_WEIGHTS = _Env.ligand_mpnn_weights()
PROTPARDELLE_OUTPUT_DIR = _Env.protpardelle_output_dir()
FOLDSEEK_BIN = _Env.foldseek_bin()

PROTPARDELLE_MODEL_CONFIGS = PROTPARDELLE_MODEL_PARAMS / "configs"
PROTPARDELLE_MODEL_WEIGHTS = PROTPARDELLE_MODEL_PARAMS / "weights"

PROTPARDELLE_RUNNING_CONFIGS = PACKAGE_ROOT_DIR / "configs" / "running"

__all__ = [
    "PROJECT_ROOT_DIR",
    "PACKAGE_ROOT_DIR",
    "PROTPARDELLE_MODEL_PARAMS",
    "ESMFOLD_PATH",
    "PROTEINMPNN_WEIGHTS",
    "LIGANDMPNN_WEIGHTS",
    "PROTPARDELLE_OUTPUT_DIR",
    "FOLDSEEK_BIN",
    "PROTPARDELLE_MODEL_CONFIGS",
    "PROTPARDELLE_MODEL_WEIGHTS",
    "PROTPARDELLE_RUNNING_CONFIGS",
]
