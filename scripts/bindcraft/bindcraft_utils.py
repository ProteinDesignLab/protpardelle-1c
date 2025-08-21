# Code from Bindcraft
# Script included for reference only

from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyrosetta as pr
from Bio.PDB import (
    DSSP,
    PDBParser,
    Selection,
)
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import _get_con_loss, get_dgram_bins, get_ptm, mask_loss
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from scipy.spatial import cKDTree


# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, "r") as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END", "LINK"))
        ]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


three_to_one_map = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
one_to_three_map = {v: k for k, v in three_to_one_map.items()}


# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]["A"], "A")
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


# Rosetta interface scores
def score_interface(pdb_file, binder_chain="B"):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []

    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set("ACFILMPVWY")
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value  # shape complementarity
    interface_interface_hbonds = (
        interfacescore.interface_hbonds
    )  # number of interface H-bonds
    interface_dG = iam.get_interface_dG()  # interface dG
    interface_dSASA = (
        iam.get_interface_delta_sasa()
    )  # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat()  # interface pack stat score
    interface_dG_SASA_ratio = (
        interfacescore.dG_dSASA_ratio * 100
    )  # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (
            interface_interface_hbonds / interface_nres
        ) * 100  # Hbonds per interface size percentage
        interface_bunsch_percentage = (
            interface_delta_unsat_hbonds / interface_nres
        ) * 100  # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0

    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if (
                res.is_apolar() == True
                or res.name() == "PHE"
                or res.name() == "TRP"
                or res.name() == "TYR"
            ):
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count / total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }

    # round to two decimal places
    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


# From https://github.com/nrbennet/dl_binder_design/blob/cafa3853ac94dceb1b908c8d9e6954d71749871a/mpnn_fr/dl_interface_design.py#L106
def thread_new_seq(pose, binder_seq):
    """
    Thread the binder sequence onto the pose being designed
    """
    rsd_set = pose.residue_type_set_for_pose(pr.rosetta.core.chemical.FULL_ATOM_t)

    for resi, mut_to in enumerate(binder_seq):
        resi += 1  # 1 indexing
        name3 = one_to_three_map[mut_to]
        new_res = pr.rosetta.core.conformation.ResidueFactory.create_residue(
            rsd_set.name_map(name3)
        )
        pose.replace_residue(resi, new_res, True)


# Relax designed structure
def pr_relax(pdb_file, relaxed_pdb_path, new_seq: str = None):
    # Generate pose
    pose = pr.pose_from_pdb(pdb_file)

    # thread in a new binder sequence
    if new_seq is not None:
        thread_new_seq(pose, new_seq)

    start_pose = pose.clone()

    ### Generate movemaps
    mmf = MoveMap()
    mmf.set_chi(True)  # enable sidechain movement
    mmf.set_bb(
        True
    )  # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
    mmf.set_jump(False)  # disable whole chain movement

    # Run FastRelax
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf)  # set MoveMap
    fastrelax.max_iter(200)  # default iterations is 2500
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)  #! try False
    fastrelax.apply(pose)

    # Align relaxed structure to original trajectory
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)

    # Copy B factors from start_pose to pose
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            # Get the B factor of the first heavy atom in the residue
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)

    # output relaxed and aligned PDB
    pose.dump_pdb(relaxed_pdb_path)
    clean_pdb(relaxed_pdb_path)


# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]), 2)


# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    """add radius of gyration loss"""

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight


# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # define distogram
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        mask_2d = np.outer(
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
        )

        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (
                    jnp.diagonal(mask_2d, 3).sum() + 1e-8
                )
        else:
            mask = offset == 3
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix": helix_loss}

    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight


def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100, 2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100, 2) if total > 0 else 0
    loop_percentage = (
        round(((total - helix - sheet) / total) * 100, 2) if total > 0 else 0
    )

    return helix_percentage, sheet_percentage, loop_percentage


# calculate secondary structure percentage of design
def calc_ss_percentage(pdb_file, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp_path = Path(__file__).parent / "dssp"
    dssp = DSSP(model, pdb_file, dssp=dssp_path)

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(
        hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys()
    )

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = "loop"
            if ss in ["H", "G", "I"]:
                ss_type = "helix"
            elif ss == "E":
                ss_type = "sheet"

            ss_counts[ss_type] += 1

            if ss_type != "loop":
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(
        total_residues, ss_counts["helix"], ss_counts["sheet"]
    )
    interface_percentages = calculate_percentages(
        total_interface_residues,
        ss_interface_counts["helix"],
        ss_interface_counts["sheet"],
    )

    i_plddt = (
        round(sum(plddts_interface) / len(plddts_interface) / 100, 2)
        if plddts_interface
        else 0
    )
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)
