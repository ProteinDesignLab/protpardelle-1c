"""Tests for protpardelle.common.protein module."""

import numpy as np
import pytest
import torch

from protpardelle.common.protein import Hetero, Protein, to_pdb


class TestProtein:
    """Test the Protein dataclass."""

    def test_protein_creation(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_chain_index,
        sample_cyclic_mask,
    ):
        """Test creating a Protein instance."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=sample_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=sample_chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        assert protein.atom_positions.shape == (L, 37, 3)
        assert protein.aatype.shape == (L,)
        assert protein.atom_mask.shape == (L, 37)
        assert protein.residue_index.shape == (L,)
        assert protein.chain_index.shape == (L,)
        assert protein.cyclic_mask.shape == (L,)
        assert protein.b_factors.shape == (L, 37)

    def test_protein_validation(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_chain_index,
        sample_cyclic_mask,
    ):
        """Test Protein validation with too many chains."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        # Create chain_index with too many chains (more than 62)
        # Ensure we have at least 63 residues to create 63 unique chain indices
        if L < 63:
            # Extend the arrays to have at least 63 residues
            extended_coords = torch.cat([sample_coords, sample_coords[: 63 - L]], dim=0)
            extended_aatype = torch.cat([sample_aatype, sample_aatype[: 63 - L]], dim=0)
            extended_mask = torch.cat(
                [sample_atom_mask, sample_atom_mask[: 63 - L]], dim=0
            )
            extended_residx = torch.cat(
                [sample_residue_index, sample_residue_index[: 63 - L]], dim=0
            )
            extended_cyclic = torch.cat(
                [sample_cyclic_mask, sample_cyclic_mask[: 63 - L]], dim=0
            )
            extended_bfactors = torch.cat([b_factors, b_factors[: 63 - L]], dim=0)
            too_many_chains = torch.arange(63, dtype=torch.long)  # 63 chains (0-62)
        else:
            extended_coords = sample_coords
            extended_aatype = sample_aatype
            extended_mask = sample_atom_mask
            extended_residx = sample_residue_index
            extended_cyclic = sample_cyclic_mask
            extended_bfactors = b_factors
            too_many_chains = torch.arange(L, dtype=torch.long) % 63

        with pytest.raises(
            ValueError, match="Cannot build an instance with more than 62 chains"
        ):
            Protein(
                atom_positions=extended_coords.numpy(),
                aatype=extended_aatype.numpy(),
                atom_mask=extended_mask.numpy(),
                residue_index=extended_residx.numpy(),
                chain_index=too_many_chains.numpy(),
                cyclic_mask=extended_cyclic.numpy(),
                b_factors=extended_bfactors.numpy(),
            )

    def test_protein_single_chain(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_cyclic_mask,
    ):
        """Test Protein with single chain."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0
        chain_index = torch.zeros(L, dtype=torch.long)

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=sample_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        assert np.all(protein.chain_index == 0)

    def test_protein_multiple_chains(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_cyclic_mask,
    ):
        """Test Protein with multiple chains."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        # Create two chains
        chain_index = torch.zeros(L, dtype=torch.long)
        chain_index[L // 2 :] = 1

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=sample_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        assert len(np.unique(protein.chain_index)) == 2


class TestHetero:
    """Test the Hetero dataclass."""

    def test_hetero_creation(self):
        """Test creating a Hetero instance."""
        # Create sample heteroatom data
        hetero_atom_positions = [
            [
                np.random.randn(5, 3) for _ in range(3)
            ],  # 3 heteroatoms with 5 atoms each
        ]
        hetero_aatype = ["ATP", "GDP", "ZN"]
        hetero_atom_types = [
            ["P", "O", "C", "N", "H"],  # ATP atoms
            ["P", "O", "C", "N", "H"],  # GDP atoms
            ["ZN"],  # ZN atom
        ]
        hetero_motif_mask = [0, 1]  # First two are motif positions
        hetero_not_motif_mask = [2]  # Last one is non-motif

        hetero = Hetero(
            hetero_atom_positions=hetero_atom_positions,
            hetero_aatype=hetero_aatype,
            hetero_atom_types=hetero_atom_types,
            hetero_motif_mask=hetero_motif_mask,
            hetero_not_motif_mask=hetero_not_motif_mask,
        )

        assert len(hetero.hetero_atom_positions) == 1  # One batch
        assert len(hetero.hetero_aatype) == 3
        assert len(hetero.hetero_atom_types) == 3
        assert len(hetero.hetero_motif_mask) == 2
        assert len(hetero.hetero_not_motif_mask) == 1


class TestToPdb:
    """Test the to_pdb function."""

    def test_to_pdb_basic(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_chain_index,
        sample_cyclic_mask,
    ):
        """Test basic PDB conversion."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=sample_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=sample_chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        pdb_str = to_pdb(protein)

        assert isinstance(pdb_str, str)
        assert "MODEL" in pdb_str
        assert "ENDMDL" in pdb_str
        assert "END" in pdb_str
        assert "ATOM" in pdb_str

    def test_to_pdb_with_chain_mapping(
        self,
        sample_coords,
        sample_aatype,
        sample_atom_mask,
        sample_residue_index,
        sample_chain_index,
        sample_cyclic_mask,
    ):
        """Test PDB conversion with custom chain ID mapping."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=sample_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=sample_chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        chain_id_mapping = {"A": 0, "B": 1}
        pdb_str = to_pdb(protein, chain_id_mapping=chain_id_mapping)

        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str

    def test_to_pdb_invalid_aatypes(
        self,
        sample_coords,
        sample_atom_mask,
        sample_residue_index,
        sample_chain_index,
        sample_cyclic_mask,
    ):
        """Test PDB conversion with invalid amino acid types."""
        L = sample_coords.shape[0]
        b_factors = torch.ones(L, 37) * 20.0

        # Create invalid aatype (values > 20)
        invalid_aatype = torch.full((L,), 25, dtype=torch.long)

        protein = Protein(
            atom_positions=sample_coords.numpy(),
            aatype=invalid_aatype.numpy(),
            atom_mask=sample_atom_mask.numpy(),
            residue_index=sample_residue_index.numpy(),
            chain_index=sample_chain_index.numpy(),
            cyclic_mask=sample_cyclic_mask.numpy(),
            b_factors=b_factors.numpy(),
        )

        with pytest.raises(ValueError, match="Invalid aatypes"):
            to_pdb(protein)
