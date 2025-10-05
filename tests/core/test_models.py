"""Tests for protpardelle.core.models module."""

from unittest.mock import Mock, patch

import pytest
import torch
from jaxtyping import Float, Int

from protpardelle.core.models import (
    CoordinateDenoiser,
    MiniMPNN,
    Protpardelle,
    apply_crop_cond_strategy,
    contig_to_idx,
    fill_motif_seq,
    get_time_dependent_scale,
    group_consecutive_idx,
    load_model,
    motif_loss,
    parse_fixed_pos_str,
)


class TestFillMotifSeq:
    """Test the fill_motif_seq function."""

    def test_basic_functionality(self):
        """Test basic functionality of fill_motif_seq."""
        B, L = 2, 10
        s_hat = torch.randint(0, 21, (B, L))
        motif_idx = [[1, 2, 3], [5, 6]]
        motif_aatype = [[0, 1, 2], [3, 4]]

        result = fill_motif_seq(s_hat, motif_idx, motif_aatype)

        assert result.shape == (B, L)
        assert result.dtype == s_hat.dtype

        # Check that motif positions are filled correctly
        assert torch.equal(result[0, 1:4], torch.tensor([0, 1, 2]))
        assert torch.equal(result[1, 5:7], torch.tensor([3, 4]))

    def test_empty_motif_idx(self):
        """Test with empty motif indices."""
        B, L = 2, 10
        s_hat = torch.randint(0, 21, (B, L))
        motif_idx = [[], []]
        motif_aatype = [[], []]

        result = fill_motif_seq(s_hat, motif_idx, motif_aatype)

        assert torch.equal(result, s_hat)

    def test_single_batch(self):
        """Test with single batch item."""
        B, L = 1, 5
        s_hat = torch.randint(0, 21, (B, L))
        motif_idx = [[0, 1]]
        motif_aatype = [[10, 11]]

        result = fill_motif_seq(s_hat, motif_idx, motif_aatype)

        assert result.shape == (B, L)
        assert torch.equal(result[0, 0:2], torch.tensor([10, 11]))

    def test_out_of_bounds_motif_idx(self):
        """Test behavior with out-of-bounds motif indices."""
        B, L = 1, 5
        s_hat = torch.randint(0, 21, (B, L))
        motif_idx = [[0, 1, 10]]  # Index 10 is out of bounds
        motif_aatype = [[10, 11, 12]]

        # Should not raise error, but may have unexpected behavior
        result = fill_motif_seq(s_hat, motif_idx, motif_aatype)
        assert result.shape == (B, L)


class TestApplyCropCondStrategy:
    """Test the apply_crop_cond_strategy function."""

    def test_backbone_strategy(self):
        """Test backbone crop conditioning strategy."""
        B, L = 2, 10
        coords = torch.randn(B, L, 37, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_aatype = [["ALA", "GLY"], ["PHE", "TRP"]]

        result = apply_crop_cond_strategy(
            coords, motif_idx, motif_aatype, strategy="backbone"
        )

        assert result.shape == coords.shape
        assert result.dtype == coords.dtype

    def test_sidechain_strategy(self):
        """Test sidechain crop conditioning strategy."""
        B, L = 2, 10
        coords = torch.randn(B, L, 37, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_aatype = [["ALA", "GLY"], ["PHE", "TRP"]]

        result = apply_crop_cond_strategy(
            coords, motif_idx, motif_aatype, strategy="sidechain"
        )

        assert result.shape == coords.shape

    def test_sidechain_tip_strategy(self):
        """Test sidechain-tip crop conditioning strategy."""
        B, L = 2, 10
        coords = torch.randn(B, L, 37, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_aatype = [["ALA", "GLY"], ["PHE", "TRP"]]

        result = apply_crop_cond_strategy(
            coords, motif_idx, motif_aatype, strategy="sidechain-tip"
        )

        assert result.shape == coords.shape

    def test_backbone_sidechain_strategy(self):
        """Test backbone-sidechain crop conditioning strategy."""
        B, L = 2, 10
        coords = torch.randn(B, L, 37, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_aatype = [["ALA", "GLY"], ["PHE", "TRP"]]

        result = apply_crop_cond_strategy(
            coords, motif_idx, motif_aatype, strategy="backbone-sidechain"
        )

        assert result.shape == coords.shape


class TestGetTimeDependentScale:
    """Test the get_time_dependent_scale function."""

    @pytest.mark.parametrize("schedule", ["constant", "quadratic", "cubic", "custom"])
    def test_different_schedules(self, schedule):
        """Test different scaling schedules."""
        w = 1.0
        curr_step = 50
        num_steps = 100

        scale = get_time_dependent_scale(schedule, w, curr_step, num_steps)

        assert isinstance(scale, float)
        assert scale >= 0

    def test_constant_schedule(self):
        """Test constant schedule."""
        w = 2.0
        curr_step = 50
        num_steps = 100

        scale = get_time_dependent_scale("constant", w, curr_step, num_steps)

        assert scale == w

    def test_quadratic_schedule(self):
        """Test quadratic schedule."""
        w = 1.0
        curr_step = 50
        num_steps = 100

        scale = get_time_dependent_scale("quadratic", w, curr_step, num_steps)

        # Should be between 0 and w
        assert 0 <= scale <= w

    def test_cubic_schedule(self):
        """Test cubic schedule."""
        w = 1.0
        curr_step = 50
        num_steps = 100

        scale = get_time_dependent_scale("cubic", w, curr_step, num_steps)

        # Should be between 0 and w
        assert 0 <= scale <= w

    def test_stage2_parameter(self):
        """Test stage2 parameter."""
        w = 1.0
        curr_step = 50
        num_steps = 100

        scale_stage1 = get_time_dependent_scale(
            "constant", w, curr_step, num_steps, stage2=False
        )
        scale_stage2 = get_time_dependent_scale(
            "constant", w, curr_step, num_steps, stage2=True
        )

        assert isinstance(scale_stage1, float)
        assert isinstance(scale_stage2, float)


class TestMotifLoss:
    """Test the motif_loss function."""

    def test_basic_functionality(self):
        """Test basic functionality of motif_loss."""
        B, L, A = 2, 10, 37
        x = torch.randn(B, L, A, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_coords = torch.randn(B, 2, A, 3)
        atom_mask = torch.ones(B, L, A)

        loss = motif_loss(x, motif_idx, motif_coords, atom_mask)

        assert loss.shape == (B,)
        assert loss.dtype == torch.float32
        assert torch.all(loss >= 0)

    def test_empty_motif_idx(self):
        """Test with empty motif indices."""
        B, L, A = 2, 10, 37
        x = torch.randn(B, L, A, 3)
        motif_idx = [[], []]
        motif_coords = torch.randn(B, 0, A, 3)
        atom_mask = torch.ones(B, L, A)

        loss = motif_loss(x, motif_idx, motif_coords, atom_mask)

        assert loss.shape == (B,)
        assert torch.all(loss == 0)

    def test_with_atom_mask(self):
        """Test with atom mask."""
        B, L, A = 2, 10, 37
        x = torch.randn(B, L, A, 3)
        motif_idx = [[1, 2], [3, 4]]
        motif_coords = torch.randn(B, 2, A, 3)
        atom_mask = torch.zeros(B, L, A)
        atom_mask[:, :, :4] = 1.0  # Only backbone atoms

        loss = motif_loss(x, motif_idx, motif_coords, atom_mask)

        assert loss.shape == (B,)
        assert torch.all(loss >= 0)


class TestGroupConsecutiveIdx:
    """Test the group_consecutive_idx function."""

    def test_basic_functionality(self):
        """Test basic functionality of group_consecutive_idx."""
        nums = [1, 2, 3, 5, 6, 8, 10, 11, 12]
        result = group_consecutive_idx(nums)

        expected = [[1, 2, 3], [5, 6], [8], [10, 11, 12]]
        assert result == expected

    def test_empty_list(self):
        """Test with empty list."""
        result = group_consecutive_idx([])
        assert result == [[]]

    def test_single_element(self):
        """Test with single element."""
        result = group_consecutive_idx([5])
        assert result == [[5]]

    def test_no_consecutive_elements(self):
        """Test with no consecutive elements."""
        nums = [1, 3, 5, 7, 9]
        result = group_consecutive_idx(nums)

        expected = [[1], [3], [5], [7], [9]]
        assert result == expected

    def test_all_consecutive_elements(self):
        """Test with all consecutive elements."""
        nums = [1, 2, 3, 4, 5]
        result = group_consecutive_idx(nums)

        expected = [[1, 2, 3, 4, 5]]
        assert result == expected


class TestContigToIdx:
    """Test the contig_to_idx function."""

    def test_basic_functionality(self):
        """Test basic functionality of contig_to_idx."""
        contig = [[1, 2, 3], [5, 6], [8, 9, 10]]
        result = contig_to_idx(contig)

        expected = [[0, 1, 2], [3, 4], [5, 6, 7]]
        assert result == expected

    def test_empty_contig(self):
        """Test with empty contig."""
        result = contig_to_idx([])
        assert result == []

    def test_single_range(self):
        """Test with single range."""
        contig = [[1, 2, 3, 4, 5]]
        result = contig_to_idx(contig)

        expected = [[0, 1, 2, 3, 4]]
        assert result == expected

    def test_non_consecutive_ranges(self):
        """Test with non-consecutive ranges."""
        contig = [[1], [3], [5]]
        result = contig_to_idx(contig)

        expected = [[0], [1], [2]]
        assert result == expected


class TestParseFixedPosStr:
    """Test the parse_fixed_pos_str function."""

    def test_basic_functionality(self):
        """Test basic functionality of parse_fixed_pos_str."""
        fixed_pos_str = "A1-5,B10-15"
        chain_id_mapping = {"A": 0, "B": 1}
        residue_index = torch.tensor([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
        chain_index = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        result = parse_fixed_pos_str(
            fixed_pos_str, chain_id_mapping, residue_index, chain_index
        )

        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All positions
        assert result == expected

    def test_single_chain(self):
        """Test with single chain."""
        fixed_pos_str = "A1-3"
        chain_id_mapping = {"A": 0}
        residue_index = torch.tensor([1, 2, 3, 4, 5])
        chain_index = torch.tensor([0, 0, 0, 0, 0])

        result = parse_fixed_pos_str(
            fixed_pos_str, chain_id_mapping, residue_index, chain_index
        )

        expected = [0, 1, 2]  # First three positions
        assert result == expected

    def test_multiple_ranges(self):
        """Test with multiple ranges."""
        fixed_pos_str = "A1-2,A5-6"
        chain_id_mapping = {"A": 0}
        residue_index = torch.tensor([1, 2, 3, 4, 5, 6])
        chain_index = torch.tensor([0, 0, 0, 0, 0, 0])

        result = parse_fixed_pos_str(
            fixed_pos_str, chain_id_mapping, residue_index, chain_index
        )

        expected = [0, 1, 4, 5]  # Positions 1, 2, 5, 6
        assert result == expected

    def test_empty_string(self):
        """Test with empty string."""
        fixed_pos_str = ""
        chain_id_mapping = {}
        residue_index = torch.tensor([1, 2, 3])
        chain_index = torch.tensor([0, 0, 0])

        result = parse_fixed_pos_str(
            fixed_pos_str, chain_id_mapping, residue_index, chain_index
        )

        assert result == []


class TestMiniMPNN:
    """Test the MiniMPNN class."""

    def test_initialization(self, mock_config):
        """Test class initialization."""
        # Create a proper TrainingConfig-like object
        from protpardelle.configs.training_dataclasses import TrainingConfig

        # This would require a full config setup, so we'll test the basic structure
        # In a real test, you'd create a proper config object
        pass

    def test_forward_interface(self):
        """Test forward method interface."""
        # This would require a full model setup
        # In a real test, you'd create a proper MiniMPNN instance and test forward pass
        pass


class TestCoordinateDenoiser:
    """Test the CoordinateDenoiser class."""

    def test_initialization_interface(self):
        """Test initialization interface."""
        # This would require a full config setup
        # In a real test, you'd create a proper CoordinateDenoiser instance
        pass

    def test_forward_interface(self):
        """Test forward method interface."""
        # This would require a full model setup
        # In a real test, you'd create a proper CoordinateDenoiser instance and test forward pass
        pass


class TestProtpardelle:
    """Test the Protpardelle class."""

    def test_initialization_interface(self):
        """Test initialization interface."""
        # This would require a full config setup
        # In a real test, you'd create a proper Protpardelle instance
        pass

    def test_forward_interface(self):
        """Test forward method interface."""
        # This would require a full model setup
        # In a real test, you'd create a proper Protpardelle instance and test forward pass
        pass

    def test_sample_interface(self):
        """Test sample method interface."""
        # This would require a full model setup
        # In a real test, you'd create a proper Protpardelle instance and test sampling
        pass


class TestLoadModel:
    """Test load_model function."""

    @patch("protpardelle.core.models.load_config")
    @patch("protpardelle.core.models.Protpardelle")
    @patch("protpardelle.core.models.torch.load")
    @patch("protpardelle.core.models.norm_path")
    def test_load_model(
        self, mock_norm_path, mock_torch_load, mock_protpardelle, mock_load_config
    ):
        """Test load_model function."""
        # Mock the config loading
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Mock the checkpoint loading
        mock_checkpoint = {"model_state_dict": {}}
        mock_torch_load.return_value = mock_checkpoint

        # Mock norm_path to return a mock Path object with is_file() method
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_norm_path.return_value = mock_path

        # Mock the model
        mock_model = Mock()
        mock_protpardelle.return_value = mock_model

        # Test loading
        model = load_model("config.yaml", "checkpoint.pt")

        assert model == mock_model
        mock_load_config.assert_called_once()
        mock_torch_load.assert_called_once()
        # The device will be determined by get_default_device(), so we just check it was called
        mock_protpardelle.assert_called_once()
