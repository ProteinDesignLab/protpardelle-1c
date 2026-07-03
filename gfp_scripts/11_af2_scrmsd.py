from pathlib import Path
import shutil

from Bio import PDB
from Bio.PDB import Superimposer
import pandas as pd
from tqdm import tqdm


def compute_backbone_rmsd(pred_pdb_path, orig_pdb_path, motif_positions=None):
    """
    Compute backbone atom RMSD between predicted and original structures.
    
    Args:
        pred_pdb_path: Path to predicted structure PDB file
        orig_pdb_path: Path to original/reference structure PDB file
        motif_positions: List of residue positions (1-indexed) for motif RMSD.
                        If None, only global RMSD is computed.
    
    Returns:
        dict with keys:
            'global_rmsd': RMSD across all backbone atoms
            'motif_rmsd': RMSD across motif backbone atoms (if motif_positions provided)
    """
    parser = PDB.PDBParser(QUIET=True)
    
    # Load structures
    pred_struct = parser.get_structure('pred', pred_pdb_path)
    orig_struct = parser.get_structure('orig', orig_pdb_path)
    
    # Get first model from each structure
    pred_model = pred_struct[0]
    orig_model = orig_struct[0]
    
    # Define backbone atoms
    backbone_atoms = ['N', 'CA', 'C', 'O']
    
    def get_backbone_atoms(model, residue_positions=None):
        """Extract backbone atoms from model, optionally filtered by positions."""
        atoms = []
        for chain in model:
            for residue in chain:
                # Skip hetero residues
                if residue.id[0] != ' ':
                    continue
                    
                res_num = residue.id[1]
                
                # Filter by position if specified
                if residue_positions is not None and res_num not in residue_positions:
                    continue
                
                # Get backbone atoms
                for atom_name in backbone_atoms:
                    if atom_name in residue:
                        atoms.append(residue[atom_name])
        return atoms
    
    def get_all_atoms(model, residue_positions=None):
        """Extract all atoms from model, optionally filtered by positions."""
        atoms = []
        for chain in model:
            for residue in chain:
                # Skip hetero residues
                if residue.id[0] != ' ':
                    continue
                    
                res_num = residue.id[1]
                
                # Filter by position if specified
                if residue_positions is not None and res_num not in residue_positions:
                    continue
                
                # Get all atoms
                for atom in residue:
                    atoms.append(atom)
        return atoms
    
    # Compute global RMSD
    pred_atoms_global = get_backbone_atoms(pred_model)
    orig_atoms_global = get_backbone_atoms(orig_model)
    
    # Ensure same number of atoms
    min_len = min(len(pred_atoms_global), len(orig_atoms_global))
    pred_atoms_global = pred_atoms_global[:min_len]
    orig_atoms_global = orig_atoms_global[:min_len]
    
    super_imposer = Superimposer()
    super_imposer.set_atoms(orig_atoms_global, pred_atoms_global)
    global_rmsd = super_imposer.rms
    
    results = {'global_rmsd': global_rmsd}
    
    # Compute motif RMSD if positions provided
    if motif_positions is not None and len(motif_positions) > 0:
        pred_atoms_motif = get_all_atoms(pred_model, set(motif_positions))
        orig_atoms_motif = get_all_atoms(orig_model, set(motif_positions))
        
        # Ensure same number of atoms
        min_len_motif = min(len(pred_atoms_motif), len(orig_atoms_motif))
        pred_atoms_motif = pred_atoms_motif[:min_len_motif]
        orig_atoms_motif = orig_atoms_motif[:min_len_motif]
        
        if len(pred_atoms_motif) > 0:
            super_imposer_motif = Superimposer()
            super_imposer_motif.set_atoms(orig_atoms_motif, pred_atoms_motif)
            results['motif_rmsd'] = super_imposer_motif.rms
        else:
            results['motif_rmsd'] = None
    
    return results


# data_dir = Path("/scratch/users/tianyulu/allatom-design-sampling-results/gfp_ensemble_design/af2/af2_pred")
# data_dir = Path("/scratch/users/tianyulu/allatom-design-sampling-results/pre_cyclic_only_gfp_ensemble_design/af2/af2_pred")
data_dir = Path("/scratch/users/tianyulu/allatom-design-sampling-results/post_cyclic_only_gfp_ensemble_design/af2/af2_pred")

all_dfs = []

for af2_output_fp in data_dir.rglob("*alphafold_results_all.csv"):
    af2_output_df = pd.read_csv(af2_output_fp)

    all_dfs.append(af2_output_df)

full_df = pd.concat(all_dfs, ignore_index=True)

# sort by pLDDT
full_df = full_df.sort_values(by="plddt", ascending=False)

full_df.to_csv(data_dir / "af2_pred_alphafold_results_all_combined.csv", index=False)

# only keep pLDDT > 0.70
plddt_threshold = 0.70
print(f"Total designs: {len(full_df)}")
full_df_filtered = full_df[full_df["plddt"] >= plddt_threshold]
print(f"Designs with pLDDT >= {plddt_threshold}: {len(full_df_filtered)}")

filtered_save_fp = data_dir / f"af2_pred_alphafold_results_all_combined_plddt_gt_{int(plddt_threshold*100)}.csv"
full_df_filtered.to_csv(filtered_save_fp, index=False)
print(f"Filtered results saved to {filtered_save_fp}")

input_data_fp_fixed = Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_fixed/chroma").glob("*with_pdb_key.csv")
input_data_fp_flex = Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_flex/chroma").glob("*with_pdb_key.csv")

input_data_df_fixed = pd.concat([pd.read_csv(fp) for fp in input_data_fp_fixed])
input_data_df_flex = pd.concat([pd.read_csv(fp) for fp in input_data_fp_flex])
input_data_df = pd.concat([input_data_df_fixed, input_data_df_flex])

pdb_key_to_fixpos = dict(zip(input_data_df["pdb_key"], input_data_df["orig_fix_pos"]))
pdb_key_to_orig_struct_fp = dict(zip(input_data_df["pdb_key"], input_data_df["chroma_repack_pdb_path"]))

global_rmsds = []
motif_rmsds = []

for example_id, af2_pdb_path in zip(full_df_filtered["example_id"], full_df_filtered["af2_pred_pdb"]):
    example_id = "_".join(example_id.split("_")[:-1])  # remove any suffix after underscore
    example_id = example_id[7:]
    try:
        fix_pos = pdb_key_to_fixpos[example_id]
    except KeyError:
        import ipdb; ipdb.set_trace()

    fix_pos_list = [int(x) for x in fix_pos.split(" ")]
    orig_struct_fp = pdb_key_to_orig_struct_fp[example_id]
    
    # compute global and motif RMSD between af2_pdb_path and orig_struct_fp
    rmsd_results = compute_backbone_rmsd(af2_pdb_path, orig_struct_fp, fix_pos_list)

    print(f"Example {example_id}:")
    print(f"  Global RMSD: {rmsd_results['global_rmsd']:.3f} Å")
    if 'motif_rmsd' in rmsd_results and rmsd_results['motif_rmsd'] is not None:
        print(f"  Motif RMSD: {rmsd_results['motif_rmsd']:.3f} Å")

    global_rmsds.append(rmsd_results['global_rmsd'])
    motif_rmsds.append(rmsd_results['motif_rmsd'])
    
full_df_filtered['af2_global_rmsd'] = global_rmsds
full_df_filtered['af2_motif_rmsd'] = motif_rmsds

full_df_filtered.to_csv(data_dir / f"af2_pred_alphafold_results_all_combined_plddt_gt_{int(plddt_threshold*100)}_with_rmsd.csv", index=False)

# filter to motif RMSD < 1.0 Å
motif_rmsd_threshold = 1.0
full_df_final = full_df_filtered[full_df_filtered['af2_motif_rmsd'] < motif_rmsd_threshold]
full_df_final.to_csv(data_dir / f"af2_pred_alphafold_results_all_combined_plddt_gt_{int(plddt_threshold*100)}_motif_rmsd_lt_{motif_rmsd_threshold}.csv", index=False)

# copy all passing structures to a separate folder
final_output_dir = data_dir / f"final_structures_plddt_gt_{int(plddt_threshold*100)}_motif_rmsd_lt_{motif_rmsd_threshold}"
final_output_dir.mkdir(exist_ok=True)
for af2_pdb_path in full_df_final["af2_pred_pdb"]:
    dest_fp = final_output_dir / Path(af2_pdb_path).name
    shutil.copy(af2_pdb_path, dest_fp)
