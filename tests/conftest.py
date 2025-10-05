"""Pytest configuration and shared fixtures for ProtPardelle tests.

This module provides common fixtures, test utilities, and configuration
that can be used across all test modules.

Author: Test Suite
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
import torch

# Set test environment variables
os.environ["PYTEST"] = "true"


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the device to use for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Return CPU device for tests that require CPU."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_pdb_file(temp_dir: Path) -> Path:
    """Create a temporary PDB file for testing."""
    pdb_content = """ATOM      1  N   ALA A   1      20.154  16.967  27.862  1.00 20.00           N
ATOM      2  CA  ALA A   1      19.030  16.067  27.862  1.00 20.00           C
ATOM      3  C   ALA A   1      17.730  16.767  27.862  1.00 20.00           C
ATOM      4  O   ALA A   1      17.530  17.967  27.862  1.00 20.00           O
END
"""
    pdb_file = temp_dir / "test.pdb"
    pdb_file.write_text(pdb_content)
    return pdb_file


@pytest.fixture
def sample_sequence() -> str:
    """Return a sample protein sequence for testing."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def sample_aatype():
    """Return a sample amino acid type tensor."""
    # Convert sequence to aatype indices (A=0, C=1, D=2, etc.)
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    aatype_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return torch.tensor([aatype_map[aa] for aa in sequence], dtype=torch.long)


@pytest.fixture
def sample_coords():
    """Return sample atomic coordinates."""
    L, A = 20, 37  # 20 residues, 37 atom types
    return torch.randn(L, A, 3)


@pytest.fixture
def sample_atom_mask():
    """Return sample atom mask."""
    L, A = 20, 37
    # Create mask where backbone atoms are present
    mask = torch.zeros(L, A)
    backbone_atoms = [0, 1, 2, 4]  # N, CA, C, O
    for i in range(L):
        mask[i, backbone_atoms] = 1.0
        # Add some sidechain atoms randomly
        if i < 15:  # Only first 15 residues have sidechains
            mask[i, 3] = 1.0  # CB
    return mask


@pytest.fixture
def sample_residue_index():
    """Return sample residue indices."""
    return torch.arange(1, 21, dtype=torch.long)  # 1-indexed like PDB


@pytest.fixture
def sample_chain_index():
    """Return sample chain indices."""
    return torch.zeros(20, dtype=torch.long)  # Single chain


@pytest.fixture
def sample_cyclic_mask():
    """Return sample cyclic mask."""
    return torch.zeros(20)  # Non-cyclic


@pytest.fixture
def sample_seq_mask():
    """Return sample sequence mask."""
    return torch.ones(20)  # All residues are valid


@pytest.fixture
def sample_noise_level():
    """Return sample noise level for diffusion."""
    return torch.tensor([0.5], dtype=torch.float32)


@pytest.fixture
def sample_timestep():
    """Return sample timestep for diffusion."""
    return torch.tensor([0.3], dtype=torch.float32)


@pytest.fixture
def sample_batch_size() -> int:
    """Return sample batch size for tests."""
    return 2


@pytest.fixture
def sample_seq_length() -> int:
    """Return sample sequence length for tests."""
    return 50


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Return a mock configuration dictionary for testing."""
    return {
        "model": {
            "task": "backbone",
            "struct_model": {
                "arch": "uvit",
                "n_atoms": 37,
                "n_channel": 6,
                "noise_cond_mult": 1,
                "uvit": {
                    "patch_size": 1,
                    "n_layers": 4,
                    "n_heads": 8,
                    "dim_head": 64,
                    "n_filt_per_layer": [64, 128, 256, 512],
                    "n_blocks_per_layer": 2,
                    "cat_pwd_to_conv": False,
                    "conv_skip_connection": True,
                    "position_embedding_type": "rotary",
                    "position_embedding_max": 32,
                    "num_cyclic_heads": 0,
                },
            },
            "mpnn_model": {
                "use_self_conditioning": False,
                "label_smoothing": 0.0,
                "n_channel": 128,
                "n_layers": 3,
                "n_neighbors": 32,
                "noise_cond_mult": 1,
            },
        },
        "data": {
            "fixed_size": 128,
            "n_aatype_tokens": 21,
            "chain_residx_gap": 200,
            "sigma_data": 10.0,
            "dummy_fill_mode": "zero",
        },
        "diffusion": {
            "training": {
                "function": "uniform",
                "psigma_mean": -0.5,
                "psigma_std": 1.5,
            },
            "sampling": {
                "function": "uniform",
                "s_min": 0.001,
                "s_max": 80.0,
            },
        },
    }


@pytest.fixture
def rng_seed() -> int:
    """Return a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(rng_seed: int) -> None:
    """Set random seeds for reproducible tests."""
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
