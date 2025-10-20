"""Tests for protpardelle.data.sequence module."""

import pytest
import torch

from protpardelle.data.sequence import seq_to_aatype, seq_to_aatype_batched


def test_sequence_module():
    """Test that sequence module can be imported."""
    from protpardelle.data import sequence

    assert sequence is not None


class TestSeqToAatype:
    """Test seq_to_aatype function."""

    def test_basic_functionality(self):
        """Test basic sequence to aatype conversion."""
        seq = "ACDEFG"
        result = seq_to_aatype(seq)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert result.shape == (6,)
        assert result.dim() == 1

    def test_different_num_tokens(self):
        """Test with different num_tokens values."""
        seq = "ACDEFG"

        # Test with 20 tokens (standard amino acids)
        result_20 = seq_to_aatype(seq, num_tokens=20)
        assert result_20.shape == (6,)
        assert result_20.dtype == torch.long

        # Test with 21 tokens (includes X)
        result_21 = seq_to_aatype(seq, num_tokens=21)
        assert result_21.shape == (6,)
        assert result_21.dtype == torch.long

        # Test with 22 tokens (includes X and mask)
        result_22 = seq_to_aatype(seq, num_tokens=22)
        assert result_22.shape == (6,)
        assert result_22.dtype == torch.long

    def test_invalid_num_tokens(self):
        """Test with invalid num_tokens value."""
        seq = "ACDEFG"

        with pytest.raises(ValueError, match="num_tokens 25 not supported"):
            seq_to_aatype(seq, num_tokens=25)

    def test_empty_sequence(self):
        """Test with empty sequence."""
        seq = ""
        result = seq_to_aatype(seq)

        assert result.shape == (0,)
        assert result.dtype == torch.long

    def test_single_amino_acid(self):
        """Test with single amino acid."""
        seq = "A"
        result = seq_to_aatype(seq)

        assert result.shape == (1,)
        assert result.dtype == torch.long

    def test_sequence_with_x(self):
        """Test sequence containing X (unknown amino acid)."""
        seq = "ACXDEF"
        result = seq_to_aatype(seq, num_tokens=21)

        assert result.shape == (6,)
        assert result.dtype == torch.long
        # X should be mapped to a specific index in the 21-token mapping

    def test_sequence_consistency(self):
        """Test that same sequence gives consistent results."""
        seq = "ACDEFGHIKL"

        result1 = seq_to_aatype(seq)
        result2 = seq_to_aatype(seq)

        assert torch.equal(result1, result2)

    def test_amino_acid_mapping(self):
        """Test that specific amino acids map to expected indices."""
        # Test that A maps to 0 in standard mapping
        seq_a = "A"
        result_a = seq_to_aatype(seq_a, num_tokens=20)
        assert result_a[0] == 0  # A should be index 0 in standard mapping


class TestSeqToAatypeBatched:
    """Test seq_to_aatype_batched function."""

    def test_basic_functionality(self):
        """Test basic batched sequence conversion."""
        seqs = ["ACDEFG", "GHIKLM"]
        result = seq_to_aatype_batched(seqs)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert result.shape == (2, 6)  # 2 sequences, max length 6

    def test_different_lengths(self):
        """Test sequences of different lengths."""
        seqs = ["AC", "ACDEFG", "A"]
        result = seq_to_aatype_batched(seqs)

        assert result.shape == (3, 6)  # 3 sequences, max length 6
        assert result.dtype == torch.long

    def test_with_max_len(self):
        """Test with specified max_len."""
        seqs = ["AC", "ACDEFG", "A"]
        result = seq_to_aatype_batched(seqs, max_len=10)

        assert result.shape == (3, 10)  # 3 sequences, max length 10
        assert result.dtype == torch.long

    def test_empty_sequences(self):
        """Test with empty sequences."""
        seqs = ["", "", ""]
        result = seq_to_aatype_batched(seqs)

        assert result.shape == (3, 0)  # 3 sequences, max length 0
        assert result.dtype == torch.long

    def test_single_sequence(self):
        """Test with single sequence."""
        seqs = ["ACDEFG"]
        result = seq_to_aatype_batched(seqs)

        assert result.shape == (1, 6)  # 1 sequence, length 6
        assert result.dtype == torch.long

    def test_padding_consistency(self):
        """Test that padding is consistent."""
        seqs = ["AC", "ACDEFG", "A"]
        result = seq_to_aatype_batched(seqs)

        # Check that shorter sequences are padded with zeros
        assert torch.all(result[0, 2:] == 0)  # First sequence padded
        assert torch.all(result[2, 1:] == 0)  # Third sequence padded

    def test_identical_sequences(self):
        """Test with identical sequences."""
        seqs = ["ACDEFG", "ACDEFG", "ACDEFG"]
        result = seq_to_aatype_batched(seqs)

        assert result.shape == (3, 6)
        # All sequences should be identical
        assert torch.equal(result[0], result[1])
        assert torch.equal(result[1], result[2])

    def test_mixed_lengths_with_padding(self):
        """Test mixed lengths with proper padding."""
        seqs = ["A", "AC", "ACDEFG", "ACDEFGHIKL"]
        result = seq_to_aatype_batched(seqs)

        assert result.shape == (4, 10)  # 4 sequences, max length 10
        assert result.dtype == torch.long

        # Check padding
        assert torch.all(result[0, 1:] == 0)  # First sequence padded
        assert torch.all(result[1, 2:] == 0)  # Second sequence padded
        assert torch.all(result[2, 6:] == 0)  # Third sequence padded
        # Fourth sequence should not be padded


class TestSequenceValidation:
    """Test sequence validation functions."""

    def test_valid_amino_acids(self):
        """Test that valid amino acids are processed correctly."""
        valid_seqs = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFG", "A"]

        for seq in valid_seqs:
            result = seq_to_aatype(seq)
            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.long
            assert result.shape == (len(seq),)

    def test_sequence_with_unknown_amino_acids(self):
        """Test sequences with unknown amino acids (X)."""
        seq_with_x = "ACXDEF"
        result = seq_to_aatype(seq_with_x, num_tokens=21)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert result.shape == (6,)

    def test_sequence_length_consistency(self):
        """Test that sequence length is preserved."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = seq_to_aatype(seq)

        assert result.shape[0] == len(seq)

    def test_batch_size_consistency(self):
        """Test that batch size is preserved in batched function."""
        seqs = ["AC", "DEF", "GHI"]
        result = seq_to_aatype_batched(seqs)

        assert result.shape[0] == len(seqs)


class TestSequenceEdgeCases:
    """Test edge cases for sequence processing."""

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        long_seq = "A" * 1000
        result = seq_to_aatype(long_seq)

        assert result.shape == (1000,)
        assert result.dtype == torch.long

    def test_sequence_with_repeated_amino_acids(self):
        """Test sequence with repeated amino acids."""
        seq = "AAAAA"
        result = seq_to_aatype(seq)

        assert result.shape == (5,)
        assert torch.all(result == result[0])  # All should be the same

    def test_mixed_case_handling(self):
        """Test that function handles case sensitivity."""
        # The function should be case-sensitive as per standard amino acid notation
        seq_upper = "ACDEFG"

        _ = seq_to_aatype(seq_upper)

        # Test that lowercase should raise an error
        with pytest.raises(KeyError):
            seq_to_aatype("acdefg")

    def test_sequence_with_special_characters(self):
        """Test sequence with special characters."""
        # This should raise an error for invalid amino acids
        seq_invalid = "AC@DEF"

        with pytest.raises(KeyError):
            seq_to_aatype(seq_invalid)

    def test_empty_batch(self):
        """Test with empty batch."""
        seqs = []

        with pytest.raises(ValueError):
            seq_to_aatype_batched(seqs)
