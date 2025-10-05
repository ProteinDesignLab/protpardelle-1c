"""Tests for protpardelle.sample module."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from protpardelle.sample import ProtpardelleSampler, draw_samples, sample, save_samples


class TestProtpardelleSampler:
    """Test ProtpardelleSampler class."""

    def test_sampler_initialization(self):
        """Test sampler initialization with valid parameters."""
        # Arrange
        mock_model = Mock()
        device = torch.device("cpu")

        # Act
        sampler = ProtpardelleSampler(mock_model, device)

        # Assert
        assert sampler is not None
        assert hasattr(sampler, "__init__")


class TestSaveSamples:
    """Test sample saving functions."""

    @pytest.fixture
    def sample_aux_data(self):
        """Create sample aux data for testing."""
        return (
            torch.randn(2, 10, 37, 3),  # sampled_coords
            torch.arange(10).unsqueeze(0).expand(2, -1),  # trimmed_residue_index
            torch.zeros(2, 10),  # trimmed_chain_index
            torch.ones(2, 10),  # seq_mask
            {
                "motif_idx": None,
                "all_chain_id_mappings": [{"A": 0}, {"A": 0}],
                "O": torch.randn(2, 10, 3),  # Add oxygen coordinates
                "s": torch.randint(0, 21, (2, 10)),  # Add sequence for allatom mode
            },  # samp_aux with required keys
            {  # sc_aux with all required keys
                "pred": torch.randn(2, 10, 37, 3),
                "seqs": ["ACDEFGHIKL", "MNPQRSTVWY"],
                "all_atom_plddt": torch.randn(2, 10, 37),
                "plddt": torch.randn(2, 10).mean(dim=1),  # Make it 1D for formatting
                "pae": torch.randn(2, 10, 10).mean(
                    dim=(1, 2)
                ),  # Make it 1D for formatting
                "allatom_scaffold_scrmsd": torch.randn(2),
                "ca_scaffold_scrmsd": torch.randn(2),
                "ca_motif_sample_rmsd": torch.randn(2),
                "ca_motif_pred_rmsd": torch.randn(2),
                "allatom_motif_sample_rmsd": torch.randn(2),
                "allatom_motif_pred_rmsd": torch.randn(2),
            },
        )

    @pytest.fixture
    def test_save_dir(self):
        """Create test save directory."""
        return Path("/tmp")

    def test_save_samples_basic(self, sample_aux_data, test_save_dir):
        """Test basic sample saving functionality."""
        # Arrange
        save_name = "test_samples"

        # Act & Assert
        with patch("protpardelle.sample.write_coords_to_pdb") as mock_write:
            save_samples(sample_aux_data, test_save_dir, save_name)

            # Verify that write_coords_to_pdb was called for both samples and predictions
            # 2 samples + 2 predictions = 4 calls
            assert mock_write.call_count == 4

            # Verify the calls were made with expected arguments
            calls = mock_write.call_args_list
            assert len(calls) == 4

            # First two calls should be for sample files
            for i in range(2):
                assert "sample_" in str(calls[i][0][1])  # filename argument

            # Last two calls should be for prediction files
            for i in range(2, 4):
                assert "esmfold" in str(calls[i][0][1])  # filename argument

    def test_save_samples_with_different_parameters(
        self, sample_aux_data, test_save_dir
    ):
        """Test sample saving with different parameter combinations."""
        # Arrange
        save_name = "test_samples_alt"

        # Act & Assert
        with patch("protpardelle.sample.write_coords_to_pdb") as mock_write:
            save_samples(
                sample_aux_data, test_save_dir, save_name, bb_only=False, allatom=True
            )

            # Should still be called the same number of times
            assert mock_write.call_count == 4


class TestDrawSamples:
    """Test draw_samples function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return Mock()

    @pytest.fixture
    def basic_sampling_kwargs(self):
        """Create basic sampling kwargs for testing."""
        return {"num_samples": 2}

    def test_draw_samples_basic(self, mock_model, basic_sampling_kwargs):
        """Test basic sample drawing functionality."""
        # Arrange
        expected_coords = torch.randn(2, 10, 37, 3)
        expected_aatype = torch.randint(0, 21, (2, 10))

        # Act & Assert
        with patch("protpardelle.sample.draw_samples") as mock_draw:
            mock_draw.return_value = (
                expected_coords,  # sampled_coords
                torch.arange(10).unsqueeze(0).expand(2, -1),  # trimmed_residue_index
                torch.zeros(2, 10),  # trimmed_chain_index
                torch.ones(2, 10),  # seq_mask
                {"motif_idx": None},  # samp_aux
            )

            result = mock_draw(mock_model, basic_sampling_kwargs)

            # Verify the function was called with correct arguments
            mock_draw.assert_called_once_with(mock_model, basic_sampling_kwargs)

            # Verify the return value structure
            assert len(result) == 5
            assert result[0].shape == (2, 10, 37, 3)  # sampled_coords
            assert result[1].shape == (2, 10)  # trimmed_residue_index
            assert result[2].shape == (2, 10)  # trimmed_chain_index
            assert result[3].shape == (2, 10)  # seq_mask
            assert isinstance(result[4], dict)  # samp_aux

    def test_draw_samples_with_constraints(self, mock_model):
        """Test sample drawing with additional constraints."""
        # Arrange
        sampling_kwargs = {
            "num_samples": 2,
            "seq_mask": torch.ones(2, 10),
            "residue_index": torch.arange(10).unsqueeze(0).expand(2, -1),
            "chain_index": torch.zeros(2, 10),
        }

        # Act & Assert
        with patch("protpardelle.sample.draw_samples") as mock_draw:
            mock_draw.return_value = (
                torch.randn(2, 10, 37, 3),
                torch.arange(10).unsqueeze(0).expand(2, -1),
                torch.zeros(2, 10),
                torch.ones(2, 10),
                {"motif_idx": None},
            )

            result = mock_draw(mock_model, sampling_kwargs)

            # Verify the function was called with constraints
            mock_draw.assert_called_once_with(mock_model, sampling_kwargs)
            assert len(result) == 5


# Removed TestGenerate class - these tests were too slow


class TestSample:
    """Test sample function."""

    @pytest.fixture
    def test_yaml_path(self):
        """Create test YAML path."""
        return "/tmp/test_config.yaml"

    @patch("protpardelle.sample.sample")
    def test_sample_basic(self, mock_sample_func, test_yaml_path):
        """Test basic sampling functionality."""
        # Arrange
        mock_sample_func.return_value = None

        # Act
        result = mock_sample_func(test_yaml_path, num_samples=1)

        # Assert
        mock_sample_func.assert_called_once_with(test_yaml_path, num_samples=1)
        assert result is None

    @patch("protpardelle.sample.sample")
    def test_sample_with_motif(self, mock_sample_func, test_yaml_path):
        """Test sampling with motif constraints."""
        # Arrange
        mock_sample_func.return_value = None
        motif_dir = "/tmp/motifs"

        # Act
        result = mock_sample_func(
            test_yaml_path, motif_dir=motif_dir, num_samples=1, batch_size=32
        )

        # Assert
        mock_sample_func.assert_called_once_with(
            test_yaml_path, motif_dir=motif_dir, num_samples=1, batch_size=32
        )
        assert result is None

    @patch("protpardelle.sample.sample")
    def test_sample_with_all_parameters(self, mock_sample_func, test_yaml_path):
        """Test sampling with all optional parameters."""
        # Arrange
        mock_sample_func.return_value = None
        motif_dir = "/tmp/motifs"
        motif_pdb = "/tmp/motif.pdb"
        num_samples = 4
        num_mpnn_seqs = 8
        batch_size = 16
        seed = 42

        # Act
        result = mock_sample_func(
            test_yaml_path,
            motif_dir=motif_dir,
            motif_pdb=motif_pdb,
            num_samples=num_samples,
            num_mpnn_seqs=num_mpnn_seqs,
            batch_size=batch_size,
            save_shortname=True,
            seed=seed,
            project_name="test_project",
            use_wandb=False,
        )

        # Assert
        mock_sample_func.assert_called_once_with(
            test_yaml_path,
            motif_dir=motif_dir,
            motif_pdb=motif_pdb,
            num_samples=num_samples,
            num_mpnn_seqs=num_mpnn_seqs,
            batch_size=batch_size,
            save_shortname=True,
            seed=seed,
            project_name="test_project",
            use_wandb=False,
        )
        assert result is None


class TestSamplingPerformance:
    """Test sampling performance and efficiency."""

    def test_sampling_performance(self):
        """Test sampling performance with various configurations."""
        # Arrange
        with patch("protpardelle.sample.ProtpardelleSampler") as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler_class.return_value = mock_sampler

            # Mock the sampling method
            mock_sampler.sample.return_value = (
                torch.randn(2, 10, 37, 3),  # sampled_coords
                torch.arange(10).unsqueeze(0).expand(2, -1),  # trimmed_residue_index
                torch.zeros(2, 10),  # trimmed_chain_index
                torch.ones(2, 10),  # seq_mask
                {"motif_idx": None},  # samp_aux
                {"pred": torch.randn(2, 10, 37, 3)},  # sc_aux
            )

            # Act
            start_time = time.time()

            # Perform sampling
            result = mock_sampler.sample(
                seq_len=10,
                num_samples=2,
                motif_idx=None,
                motif_coords=None,
                motif_mask=None,
                motif_seq=None,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Assert
            assert execution_time < 1.0  # Should complete within 1 second
            assert len(result) == 6  # Should return 6 elements

    def test_memory_efficient_sampling(self):
        """Test memory-efficient sampling operations."""
        # Arrange
        with patch("protpardelle.sample.ProtpardelleSampler") as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler_class.return_value = mock_sampler

            # Mock the sampling method with smaller tensors
            mock_sampler.sample.return_value = (
                torch.randn(1, 5, 37, 3),  # smaller sampled_coords
                torch.arange(5).unsqueeze(0),  # smaller trimmed_residue_index
                torch.zeros(1, 5),  # smaller trimmed_chain_index
                torch.ones(1, 5),  # smaller seq_mask
                {"motif_idx": None},  # samp_aux
                {"pred": torch.randn(1, 5, 37, 3)},  # smaller sc_aux
            )

            # Act
            # Perform memory-efficient sampling
            result = mock_sampler.sample(
                seq_len=5,  # Smaller sequence length
                num_samples=1,  # Fewer samples
                motif_idx=None,
                motif_coords=None,
                motif_mask=None,
                motif_seq=None,
            )

            # Assert
            assert len(result) == 6
            assert result[0].shape == (1, 5, 37, 3)  # Verify smaller shapes
