"""Tests for protpardelle.common.residue_constants module."""

import numpy as np
import pytest

from protpardelle.common import residue_constants


class TestResidueConstants:
    """Test residue constants."""

    def test_restypes(self):
        """Test that restypes contains all 20 amino acids."""
        assert len(residue_constants.restypes) == 20
        assert "A" in residue_constants.restypes
        assert "R" in residue_constants.restypes
        assert "N" in residue_constants.restypes
        assert "D" in residue_constants.restypes
        assert "C" in residue_constants.restypes
        assert "Q" in residue_constants.restypes
        assert "E" in residue_constants.restypes
        assert "G" in residue_constants.restypes
        assert "H" in residue_constants.restypes
        assert "I" in residue_constants.restypes
        assert "L" in residue_constants.restypes
        assert "K" in residue_constants.restypes
        assert "M" in residue_constants.restypes
        assert "F" in residue_constants.restypes
        assert "P" in residue_constants.restypes
        assert "S" in residue_constants.restypes
        assert "T" in residue_constants.restypes
        assert "W" in residue_constants.restypes
        assert "Y" in residue_constants.restypes
        assert "V" in residue_constants.restypes

    def test_restype_order(self):
        """Test restype_order mapping."""
        for i, restype in enumerate(residue_constants.restypes):
            assert residue_constants.restype_order[restype] == i

    def test_restype_1to3(self):
        """Test 1-letter to 3-letter amino acid mapping."""
        assert residue_constants.restype_1to3["A"] == "ALA"
        assert residue_constants.restype_1to3["R"] == "ARG"
        assert residue_constants.restype_1to3["N"] == "ASN"
        assert residue_constants.restype_1to3["D"] == "ASP"
        assert residue_constants.restype_1to3["C"] == "CYS"
        assert residue_constants.restype_1to3["Q"] == "GLN"
        assert residue_constants.restype_1to3["E"] == "GLU"
        assert residue_constants.restype_1to3["G"] == "GLY"
        assert residue_constants.restype_1to3["H"] == "HIS"
        assert residue_constants.restype_1to3["I"] == "ILE"
        assert residue_constants.restype_1to3["L"] == "LEU"
        assert residue_constants.restype_1to3["K"] == "LYS"
        assert residue_constants.restype_1to3["M"] == "MET"
        assert residue_constants.restype_1to3["F"] == "PHE"
        assert residue_constants.restype_1to3["P"] == "PRO"
        assert residue_constants.restype_1to3["S"] == "SER"
        assert residue_constants.restype_1to3["T"] == "THR"
        assert residue_constants.restype_1to3["W"] == "TRP"
        assert residue_constants.restype_1to3["Y"] == "TYR"
        assert residue_constants.restype_1to3["V"] == "VAL"

    def test_restype_3to1(self):
        """Test 3-letter to 1-letter amino acid mapping."""
        assert residue_constants.restype_3to1["ALA"] == "A"
        assert residue_constants.restype_3to1["ARG"] == "R"
        assert residue_constants.restype_3to1["ASN"] == "N"
        assert residue_constants.restype_3to1["ASP"] == "D"
        assert residue_constants.restype_3to1["CYS"] == "C"
        assert residue_constants.restype_3to1["GLN"] == "Q"
        assert residue_constants.restype_3to1["GLU"] == "E"
        assert residue_constants.restype_3to1["GLY"] == "G"
        assert residue_constants.restype_3to1["HIS"] == "H"
        assert residue_constants.restype_3to1["ILE"] == "I"
        assert residue_constants.restype_3to1["LEU"] == "L"
        assert residue_constants.restype_3to1["LYS"] == "K"
        assert residue_constants.restype_3to1["MET"] == "M"
        assert residue_constants.restype_3to1["PHE"] == "F"
        assert residue_constants.restype_3to1["PRO"] == "P"
        assert residue_constants.restype_3to1["SER"] == "S"
        assert residue_constants.restype_3to1["THR"] == "T"
        assert residue_constants.restype_3to1["TRP"] == "W"
        assert residue_constants.restype_3to1["TYR"] == "Y"
        assert residue_constants.restype_3to1["VAL"] == "V"

    def test_atom_types(self):
        """Test atom types list."""
        assert "N" in residue_constants.atom_types
        assert "CA" in residue_constants.atom_types
        assert "C" in residue_constants.atom_types
        assert "O" in residue_constants.atom_types
        assert "CB" in residue_constants.atom_types
        assert len(residue_constants.atom_types) == 37

    def test_atom_order(self):
        """Test atom order mapping."""
        for i, atom_type in enumerate(residue_constants.atom_types):
            assert residue_constants.atom_order[atom_type] == i

    def test_backbone_atoms(self):
        """Test backbone atoms list."""
        assert residue_constants.backbone_atoms == ["N", "CA", "C", "O"]

    def test_sidechain_atoms(self):
        """Test sidechain atoms list."""
        backbone_atoms = set(residue_constants.backbone_atoms)
        all_atoms = set(residue_constants.atom_types)
        sidechain_atoms = all_atoms - backbone_atoms
        assert set(residue_constants.sidechain_atoms) == sidechain_atoms

    def test_chi_angles_atoms(self):
        """Test chi angles atoms for different amino acids."""
        # Test that ALA has no chi angles
        assert residue_constants.chi_angles_atoms["ALA"] == []

        # Test that GLY has no chi angles
        assert residue_constants.chi_angles_atoms["GLY"] == []

        # Test that ARG has 4 chi angles
        assert len(residue_constants.chi_angles_atoms["ARG"]) == 4

        # Test that CYS has 1 chi angle
        assert len(residue_constants.chi_angles_atoms["CYS"]) == 1

    def test_chi_angles_mask(self):
        """Test chi angles mask."""
        assert len(residue_constants.chi_angles_mask) == 20  # 20 amino acids
        assert len(residue_constants.chi_angles_mask[0]) == 4  # 4 chi angles max

        # ALA should have all zeros (no chi angles)
        assert all(x == 0.0 for x in residue_constants.chi_angles_mask[0])

        # ARG should have all ones (4 chi angles)
        assert all(x == 1.0 for x in residue_constants.chi_angles_mask[1])

    def test_rigid_group_atom_positions(self):
        """Test rigid group atom positions."""
        # Test that all amino acids have rigid group positions
        for restype in residue_constants.restypes:
            resname = residue_constants.restype_1to3[restype]
            assert resname in residue_constants.rigid_group_atom_positions

        # Test that ALA has the expected atoms
        ala_positions = residue_constants.rigid_group_atom_positions["ALA"]
        atom_names = [pos[0] for pos in ala_positions]
        assert "N" in atom_names
        assert "CA" in atom_names
        assert "C" in atom_names
        assert "CB" in atom_names
        assert "O" in atom_names

    def test_residue_atoms(self):
        """Test residue atoms for different amino acids."""
        # Test ALA atoms
        ala_atoms = residue_constants.residue_atoms["ALA"]
        assert "N" in ala_atoms
        assert "CA" in ala_atoms
        assert "C" in ala_atoms
        assert "CB" in ala_atoms
        assert "O" in ala_atoms

        # Test GLY atoms (no CB)
        gly_atoms = residue_constants.residue_atoms["GLY"]
        assert "N" in gly_atoms
        assert "CA" in gly_atoms
        assert "C" in gly_atoms
        assert "O" in gly_atoms
        assert "CB" not in gly_atoms

    def test_sequence_to_onehot(self):
        """Test sequence to onehot conversion."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        mapping = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        onehot = residue_constants.sequence_to_onehot(sequence, mapping)

        assert onehot.shape == (len(sequence), len(mapping))
        assert onehot.dtype == np.int32

        # Check that each position has exactly one 1
        for i in range(len(sequence)):
            assert onehot[i].sum() == 1
            assert onehot[i, mapping[sequence[i]]] == 1

    def test_sequence_to_onehot_with_unknown(self):
        """Test sequence to onehot with unknown amino acids."""
        sequence = "ACDEFGHIKLMNPQRSTVWYX"  # X is unknown
        mapping = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        mapping["X"] = len(mapping)  # Add X to mapping

        onehot = residue_constants.sequence_to_onehot(
            sequence, mapping, map_unknown_to_x=True
        )

        assert onehot.shape == (len(sequence), len(mapping))

        # Check that X maps to the X position
        x_pos = sequence.index("X")
        assert onehot[x_pos, mapping["X"]] == 1

    def test_standard_atom_mask(self):
        """Test standard atom mask."""
        mask = residue_constants.STANDARD_ATOM_MASK

        assert mask.shape == (21, 37)  # 20 amino acids + 1 unknown, 37 atom types
        assert mask.dtype == np.int32

        # Check that ALA has the expected atoms
        ala_idx = residue_constants.restype_order["A"]
        ala_mask = mask[ala_idx]

        # N, CA, C, CB, O should be present
        n_idx = residue_constants.atom_order["N"]
        ca_idx = residue_constants.atom_order["CA"]
        c_idx = residue_constants.atom_order["C"]
        cb_idx = residue_constants.atom_order["CB"]
        o_idx = residue_constants.atom_order["O"]

        assert ala_mask[n_idx] == 1
        assert ala_mask[ca_idx] == 1
        assert ala_mask[c_idx] == 1
        assert ala_mask[cb_idx] == 1
        assert ala_mask[o_idx] == 1

    def test_van_der_waals_radius(self):
        """Test van der Waals radius values."""
        assert "C" in residue_constants.van_der_waals_radius
        assert "N" in residue_constants.van_der_waals_radius
        assert "O" in residue_constants.van_der_waals_radius
        assert "S" in residue_constants.van_der_waals_radius

        # Check that values are positive
        for atom, radius in residue_constants.van_der_waals_radius.items():
            assert radius > 0

    def test_ca_ca_distance(self):
        """Test CA-CA distance constant."""
        assert residue_constants.ca_ca > 0
        assert isinstance(residue_constants.ca_ca, float)

    def test_between_res_bond_lengths(self):
        """Test between-residue bond lengths."""
        assert len(residue_constants.between_res_bond_length_c_n) == 2
        assert len(residue_constants.between_res_bond_length_stddev_c_n) == 2

        # Check that values are positive
        for length in residue_constants.between_res_bond_length_c_n:
            assert length > 0
        for stddev in residue_constants.between_res_bond_length_stddev_c_n:
            assert stddev > 0

    def test_between_res_cos_angles(self):
        """Test between-residue cosine angles."""
        assert len(residue_constants.between_res_cos_angles_c_n_ca) == 2
        assert len(residue_constants.between_res_cos_angles_ca_c_n) == 2

        # Check that values are in valid range [-1, 1]
        for cos_angle in residue_constants.between_res_cos_angles_c_n_ca:
            assert -1 <= cos_angle <= 1
        for cos_angle in residue_constants.between_res_cos_angles_ca_c_n:
            assert -1 <= cos_angle <= 1
