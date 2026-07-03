"""
GFP Scaffold Filter and Motif Alignment Script

This script:
1. Loads designed structures and filters by motif SASA
2. Superimposes designs to motif backbone atoms
3. Aligns reference motif structures (pre_cyclic, post_cyclic, gfp_00000-gfp_00020)
4. Replaces scaffold motif residues with aligned reference motif sidechains

The SASA threshold is derived from native GFP chromophore SASA plus a configurable buffer.
"""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import subprocess
import tempfile

from Bio.PDB import PDBIO, Atom, Chain, PDBParser, Residue, Selection, ShrakeRupley, Superimposer
import numpy as np
import pandas as pd
from tqdm import tqdm

# PULCHRA path for sidechain packing
PULCHRA_PATH = "/oak/stanford/groups/possu/tianyu/pulchra_306/pulchra"


@dataclass
class MotifConfig:
    """Configuration for a motif structure."""
    name: str
    pdb_path: Path
    # Residue indices to extract (full motif)
    full_residx: list[int]
    # Residue indices to use for alignment (excluding flexible regions)
    align_residx: list[int]


def parse_motif_indices(motif_str: str) -> list[int]:
    """
    Parse motif string like 'A58-71;A96-96;A222-222' into list of residue indices.

    Args:
        motif_str: Semicolon-separated motif ranges (e.g., 'A58-71;A96-96')

    Returns:
        List of residue indices
    """
    residx = []
    for segment in motif_str.split(";"):
        # Remove chain letter prefix
        segment = segment.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if "-" in segment:
            start, end = segment.split("-")
            residx.extend(range(int(start), int(end) + 1))
        else:
            residx.append(int(segment))
    return residx


def get_alignment_indices(full_residx: list[int], exclude_range: tuple[int, int] = None) -> list[int]:
    """
    Get indices for alignment, optionally excluding a flexible region.

    Args:
        full_residx: Full list of motif residue indices
        exclude_range: Tuple of (start, end) indices to exclude (inclusive)

    Returns:
        List of residue indices for alignment
    """
    if exclude_range is None:
        return full_residx

    start, end = exclude_range
    return [idx for idx in full_residx if idx < start or idx > end]


def compute_native_chromophore_sasa(
    native_pdb_path: Path,
    motif_residx: list[int],
    exclude_range: tuple[int, int] = None,
) -> float:
    """
    Compute SASA of the chromophore region in a native GFP structure.

    Args:
        native_pdb_path: Path to native GFP PDB file
        motif_residx: Full motif residue indices (1-indexed, as in PDB)
        exclude_range: Tuple of (start, end) indices to exclude (flexible region)

    Returns:
        Total SASA for the chromophore region (excluding flexible residues)
    """
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()

    structure = parser.get_structure("", native_pdb_path)
    sr.compute(structure, level="R")

    # Get common residues (excluding flexible region)
    if exclude_range:
        start, end = exclude_range
        common_residx = [idx for idx in motif_residx if idx < start or idx > end]
    else:
        common_residx = motif_residx

    total_sasa = sum(structure[0]['A'][ri].sasa for ri in common_residx)
    return total_sasa


def pack_sidechains_with_pulchra(pdb_path: str, motif_residue_nums: set[int], tmpdir: str, pack_residue: str = 'TYR') -> str:
    """
    Use PULCHRA to pack sidechains on non-motif GLY residues.

    Args:
        pdb_path: Path to input PDB file
        motif_residue_nums: Set of 1-indexed residue numbers to preserve (motif)
        tmpdir: Temporary directory for PULCHRA files
        pack_residue: Residue type to use for packing (default: TYR for better occlusion)

    Returns:
        Path to rebuilt PDB file, or None if failed
    """
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # Load original structure and save motif residue coordinates
    structure_orig = parser.get_structure('orig', pdb_path)
    motif_residues_orig = {}
    for chain in structure_orig[0]:
        for res in chain:
            if res.id[1] in motif_residue_nums:
                motif_residues_orig[res.id[1]] = deepcopy(res)

    # Load structure and mutate GLY to pack_residue outside motif
    structure = parser.get_structure('', pdb_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[1] not in motif_residue_nums and residue.resname == 'GLY':
                    residue.resname = pack_residue

    # Save modified structure
    tmp_pdb = Path(tmpdir) / "input.pdb"
    io.set_structure(structure)
    io.save(str(tmp_pdb))

    # Run PULCHRA
    result = subprocess.run(
        [PULCHRA_PATH, str(tmp_pdb)],
        capture_output=True, text=True, cwd=tmpdir
    )

    rebuilt_pdb = Path(tmpdir) / "input.rebuilt.pdb"
    if not rebuilt_pdb.exists():
        return None

    # Load rebuilt structure and restore original motif residues
    rebuilt_structure = parser.get_structure('', rebuilt_pdb)
    first_chain = list(rebuilt_structure[0].get_chains())[0]

    # Replace motif residues with originals (to keep original sidechains)
    for res_id, orig_res in motif_residues_orig.items():
        if res_id in [r.id[1] for r in first_chain]:
            rebuilt_res = first_chain[res_id]
            first_chain.detach_child(rebuilt_res.id)
            orig_res_copy = deepcopy(orig_res)
            orig_res_copy.id = (' ', res_id, ' ')
            first_chain.add(orig_res_copy)

    # Save final structure
    final_pdb = Path(tmpdir) / "final.pdb"
    io.set_structure(rebuilt_structure)
    io.save(str(final_pdb))

    return str(final_pdb)


def compute_sasa(pdb_path: str, residue_indices: list[int], zero_indexed: bool = True, use_pulchra: bool = True) -> float:
    """
    Compute total SASA for specified residues.

    Args:
        pdb_path: Path to PDB file
        residue_indices: List of residue indices to compute SASA for
        zero_indexed: If True, indices are 0-indexed and will be converted to
                      1-indexed PDB residue numbers by adding 1
        use_pulchra: If True, use PULCHRA to pack VAL sidechains on surrounding
                     GLY residues before computing SASA

    Returns:
        Total SASA for the specified residues
    """
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()

    # Convert 0-indexed to 1-indexed PDB residue numbers if needed
    pdb_residue_nums = [ri + 1 for ri in residue_indices] if zero_indexed else residue_indices
    pdb_residue_set = set(pdb_residue_nums)

    if use_pulchra:
        # Use PULCHRA to pack VAL sidechains on non-motif residues
        with tempfile.TemporaryDirectory() as tmpdir:
            packed_pdb = pack_sidechains_with_pulchra(pdb_path, pdb_residue_set, tmpdir)
            if packed_pdb:
                structure = parser.get_structure("", packed_pdb)
            else:
                # Fallback to original structure if PULCHRA fails
                structure = parser.get_structure("", pdb_path)
    else:
        structure = parser.get_structure("", pdb_path)

    sr.compute(structure, level="R")

    # Get first chain (PULCHRA may change chain ID)
    first_chain = list(structure[0].get_chains())[0]
    available_residues = {r.id[1] for r in first_chain}
    motif_residues_present = pdb_residue_set & available_residues

    total_sasa = sum(first_chain[ri].sasa for ri in motif_residues_present)
    return total_sasa


def _compute_sasa_worker(args: tuple) -> float:
    """
    Worker function for parallel SASA computation.

    Args:
        args: Tuple of (pdb_path, common_pos) where common_pos is 0-indexed

    Returns:
        Computed SASA value, or inf on error
    """
    pdb_path, common_pos = args
    try:
        return compute_sasa(pdb_path, common_pos, use_pulchra=True)
    except Exception:
        return float('inf')


def extract_backbone_atoms(structure, chain_id: str, residue_indices: list[int], zero_indexed: bool = True) -> list:
    """
    Extract backbone atoms (N, CA, C) from specified residues.

    Args:
        structure: BioPython structure object
        chain_id: Chain identifier
        residue_indices: List of residue indices
        zero_indexed: If True, indices are 0-indexed and will be converted to
                      1-indexed PDB residue numbers by adding 1

    Returns:
        List of backbone atoms
    """
    # Convert 0-indexed to 1-indexed PDB residue numbers if needed
    pdb_residue_nums = [ri + 1 for ri in residue_indices] if zero_indexed else residue_indices

    atoms = []
    for pos in pdb_residue_nums:
        for atom in Selection.unfold_entities(structure[0][chain_id][pos], "A"):
            if atom.name in ['N', 'CA', 'C']:
                atoms.append(atom)
    return atoms


def superimpose_structure(reference_atoms: list, moving_atoms: list, structure) -> None:
    """
    Superimpose a structure onto reference atoms in place.

    Args:
        reference_atoms: List of reference atoms
        moving_atoms: List of atoms to align
        structure: Structure to transform
    """
    sup = Superimposer()
    sup.set_atoms(reference_atoms, moving_atoms)
    sup.apply(structure)


def create_motif_chain(
    motif_residues: list,
    scaffold_fixpos: list[int],
    chain_id: str = "B",
    zero_indexed: bool = True
) -> Chain.Chain:
    """
    Create a new chain from motif residues with updated residue numbering.

    Args:
        motif_residues: List of BioPython residue objects
        scaffold_fixpos: List of residue positions in the scaffold
        chain_id: Chain identifier for the new chain
        zero_indexed: If True, scaffold_fixpos is 0-indexed and will be converted
                      to 1-indexed PDB residue numbers by adding 1

    Returns:
        New BioPython Chain object
    """
    new_chain = Chain.Chain(chain_id)

    # Convert to 1-indexed PDB residue numbers if needed
    pdb_residue_nums = [p + 1 for p in scaffold_fixpos] if zero_indexed else scaffold_fixpos

    for k, res in enumerate(motif_residues):
        new_res = Residue.Residue(
            (" ", pdb_residue_nums[k], " "), res.resname, res.segid
        )
        for atom in res.get_atoms():
            new_atom = Atom.Atom(
                atom.get_name(),
                atom.get_coord().copy(),
                atom.get_bfactor(),
                atom.get_occupancy(),
                atom.get_altloc(),
                atom.get_fullname(),
                atom.get_serial_number(),
                element=atom.element,
            )
            new_res.add(new_atom)
        new_chain.add(new_res)

    return new_chain


def replace_motif_in_scaffold(
    scaffold_structure,
    motif_chain: Chain.Chain,
    scaffold_fixpos: list[int],
    zero_indexed: bool = True
) -> None:
    """
    Replace scaffold motif residues with motif chain residues.

    Args:
        scaffold_structure: BioPython structure to modify
        motif_chain: Chain containing motif residues
        scaffold_fixpos: List of residue positions to replace
        zero_indexed: If True, scaffold_fixpos is 0-indexed and will be converted
                      to 1-indexed PDB residue numbers by adding 1
    """
    # Convert to 1-indexed PDB residue numbers if needed
    pdb_residue_nums = set(p + 1 for p in scaffold_fixpos) if zero_indexed else set(scaffold_fixpos)

    # Remove overlapping residues from chain A
    chain_a = scaffold_structure[0]["A"]
    for res in list(Selection.unfold_entities(chain_a, "R")):
        if res.id[1] in pdb_residue_nums:
            chain_a.detach_child(res.id)

    # Add motif chain and relabel to A
    scaffold_structure[0].add(motif_chain)
    motif_chain.id = "A"


def main():
    # Configuration
    MOTIF_DIR = Path("/home/users/tianyulu/code/protpardelle-1c/examples/motifs/gfp")

    # SASA threshold configuration
    # Native GFP (pre_cyclic / 2awj) has chromophore SASA of ~34 Å² (excluding flexible region)
    # Add a small buffer to allow for slight variations in designs
    SASA_BUFFER = 5.0  # Å² above native SASA

    # Input configurations: (design_input_csv, output_base_dir)
    INPUT_CONFIGS = [
        (
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_05/design_input.csv"),
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_05"),
        ),
        (
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/design_input.csv"),
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_08"),
        ),
    ]

    # Motif indices configuration (1-indexed as in PDB files)
    # Pre/post cyclic: A58-71;A96-96;A222-222
    # GFP intermediates: A56-69;A94-94;A220-220 (two less)
    PRE_POST_MOTIF_STR = "A58-71;A96-96;A222-222"
    GFP_INTERMED_MOTIF_STR = "A56-69;A94-94;A220-220"

    # Flexible region to exclude from alignment (residues 65-68 for pre/post, 63-66 for intermediates)
    PRE_POST_EXCLUDE = (65, 68)
    GFP_INTERMED_EXCLUDE = (63, 66)

    # Parse motif indices
    pre_post_full_residx = parse_motif_indices(PRE_POST_MOTIF_STR)
    pre_post_align_residx = get_alignment_indices(pre_post_full_residx, PRE_POST_EXCLUDE)

    gfp_intermed_full_residx = parse_motif_indices(GFP_INTERMED_MOTIF_STR)
    gfp_intermed_align_residx = get_alignment_indices(gfp_intermed_full_residx, GFP_INTERMED_EXCLUDE)

    # Compute native GFP chromophore SASA as reference
    # Using pre_cyclic (same as 2awj) as the reference native structure
    native_pdb = MOTIF_DIR / "pre_cyclic.pdb"
    native_sasa = compute_native_chromophore_sasa(
        native_pdb, pre_post_full_residx, PRE_POST_EXCLUDE
    )
    MAX_MOTIF_SASA = native_sasa + SASA_BUFFER
    print(f"Native GFP chromophore SASA: {native_sasa:.2f} Å²")
    print(f"SASA threshold (native + {SASA_BUFFER} buffer): {MAX_MOTIF_SASA:.2f} Å²")

    # Build motif configurations
    motif_configs = [
        MotifConfig(
            name="pre_cyclic",
            pdb_path=MOTIF_DIR / "pre_cyclic.pdb",
            full_residx=pre_post_full_residx,
            align_residx=pre_post_align_residx,
        ),
        MotifConfig(
            name="post_cyclic",
            pdb_path=MOTIF_DIR / "post_cyclic.pdb",
            full_residx=pre_post_full_residx,
            align_residx=pre_post_align_residx,
        ),
    ]

    # Add GFP intermediate structures (gfp_00020 down to gfp_00000)
    for i in range(20, -1, -1):
        motif_configs.append(MotifConfig(
            name=f"gfp_{i:05d}",
            pdb_path=MOTIF_DIR / f"gfp_{i:05d}.pdb",
            full_residx=gfp_intermed_full_residx,
            align_residx=gfp_intermed_align_residx,
        ))

    # Initialize BioPython objects
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    # Process each input configuration
    for design_input_csv, BASE_DIR in INPUT_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Processing: {design_input_csv}")
        print(f"Output dir: {BASE_DIR}")
        print(f"{'='*60}")

        if not design_input_csv.exists():
            print(f"Warning: {design_input_csv} not found, skipping...")
            continue

        # Check for cached filtered results first
        filtered_csv_path = BASE_DIR / f"design_input_sasa_filtered.csv"
        sasa_csv_path = BASE_DIR / f"design_input_with_sasa.csv"

        if filtered_csv_path.exists():
            print(f"Loading cached filtered results from: {filtered_csv_path}")
            design_df = pd.read_csv(filtered_csv_path)
            print(f"Loaded {len(design_df)} pre-filtered designs (SASA <= {MAX_MOTIF_SASA:.2f})", flush=True)

            if len(design_df) == 0:
                print("No designs in cached file, skipping...")
                continue
        else:
            # Load design input with fix_pos from cluster analysis
            design_df = pd.read_csv(design_input_csv)
            print(f"Loaded {len(design_df)} cluster representatives")

            # Compute SASA for each design using fix_pos from the CSV
            if sasa_csv_path.exists():
                print("Loading pre-computed SASA values...")
                design_df = pd.read_csv(sasa_csv_path)
            else:
                print("Computing SASA for each design (parallelized)...")

                # Prepare arguments for parallel computation
                sasa_args = []
                for _, row in design_df.iterrows():
                    pdb_path = row["pdb_path"]
                    fix_pos = [int(p) for p in row["fix_pos"].split(",")]

                    # Common positions: first 7 + positions 11 onwards (excluding flexible region 7-10)
                    # fix_pos format: 14 chromophore indices + MET index + GLU index
                    # Flexible region is indices 7-10 (0-indexed within the chromophore stretch)
                    common_pos = fix_pos[:7] + fix_pos[11:]
                    sasa_args.append((pdb_path, common_pos))

                # Parallel SASA computation
                n_workers = min(cpu_count(), 32)  # Cap at 32 workers
                print(f"Using {n_workers} workers...", flush=True)

                with Pool(n_workers) as pool:
                    all_sasa = list(tqdm(
                        pool.imap(_compute_sasa_worker, sasa_args),
                        total=len(sasa_args)
                    ))

                design_df['sasa'] = all_sasa
                design_df.to_csv(sasa_csv_path, index=False)
                print(f"Saved SASA values to: {sasa_csv_path}", flush=True)

            # Filter by SASA
            print(f"Before SASA filtering: {len(design_df)} designs", flush=True)
            design_df = design_df[design_df['sasa'] <= MAX_MOTIF_SASA].copy()
            print(f"After SASA filtering (threshold={MAX_MOTIF_SASA:.2f}): {len(design_df)} designs", flush=True)

            if len(design_df) == 0:
                print("No designs passed SASA filter, skipping...")
                continue

            # Save filtered results
            design_df.to_csv(filtered_csv_path, index=False)
            print(f"Saved filtered designs to: {filtered_csv_path}", flush=True)

        # Build mapping from pdb path to fixpos
        pdb_to_fixpos = {}
        pdb_to_common_fixpos = {}
        for _, row in design_df.iterrows():
            pdb_path = row["pdb_path"]
            fix_pos = [int(p) for p in row["fix_pos"].split(",")]
            pdb_to_fixpos[pdb_path] = fix_pos
            pdb_to_common_fixpos[pdb_path] = fix_pos[:7] + fix_pos[11:]

        # Load and align scaffold structures
        print("Loading and aligning scaffold structures...")
        scaffold_structures = []
        scaffold_motif_atoms = []
        scaffold_fixpos_list = []

        for pdb_path in tqdm(design_df["pdb_path"]):
            structure = parser.get_structure("", pdb_path)
            common_fixpos = pdb_to_common_fixpos[pdb_path]

            motif_atoms = extract_backbone_atoms(structure, "A", common_fixpos)

            scaffold_structures.append(structure)
            scaffold_motif_atoms.append(motif_atoms)
            scaffold_fixpos_list.append(pdb_to_fixpos[pdb_path])

        # Align all scaffolds to the first one
        reference_atoms = scaffold_motif_atoms[0]
        aligned_scaffolds = []

        for i, moving_atoms in enumerate(scaffold_motif_atoms):
            aligned_struct = deepcopy(scaffold_structures[i])
            superimpose_structure(reference_atoms, moving_atoms, aligned_struct)
            aligned_scaffolds.append(aligned_struct)

        # Save aligned scaffolds
        aligned_dir = BASE_DIR / "aligned"
        aligned_dir.mkdir(exist_ok=True, parents=True)

        pdb_stems = [Path(fp).stem for fp in design_df["pdb_path"]]
        for stem, structure in zip(pdb_stems, aligned_scaffolds):
            io.set_structure(structure)
            io.save(str(aligned_dir / f"{stem}.pdb"))

        # Process each motif configuration
        aligned_paths = defaultdict(list)

        for config in motif_configs:
            print(f"\nProcessing motif: {config.name}")

            # Load motif structure
            motif_structure = parser.get_structure("", config.pdb_path)

            # Extract alignment atoms from motif (using 1-indexed PDB residue numbers)
            motif_align_atoms = extract_backbone_atoms(motif_structure, "A", config.align_residx, zero_indexed=False)

            # Align motif to reference scaffold
            try:
                superimpose_structure(reference_atoms, motif_align_atoms, motif_structure)
            except Exception as e:
                print(f"Error aligning {config.name}: {e}")
                print(f"Reference atoms: {len(reference_atoms)}, Motif atoms: {len(motif_align_atoms)}")
                continue

            # Extract full motif residues after alignment
            aligned_motif_residues = [motif_structure[0]["A"][pos] for pos in config.full_residx]

            # Create output directory
            output_dir = BASE_DIR / f"{config.name}_superimposed"
            output_dir.mkdir(exist_ok=True, parents=True)

            # Apply motif to each aligned scaffold
            print(f"Superimposing {config.name} to scaffolds...")
            for j, scaffold in tqdm(enumerate(aligned_scaffolds), total=len(aligned_scaffolds)):
                scaffold_copy = deepcopy(scaffold)
                scaffold_fixpos = scaffold_fixpos_list[j]

                # Create motif chain with scaffold numbering
                motif_chain = create_motif_chain(aligned_motif_residues, scaffold_fixpos, "B")

                # Replace scaffold motif with reference motif
                replace_motif_in_scaffold(scaffold_copy, motif_chain, scaffold_fixpos)

                # Save
                save_path = output_dir / f"{pdb_stems[j]}.pdb"
                io.set_structure(scaffold_copy)
                io.save(str(save_path))

                aligned_paths[config.name].append(str(save_path))

        # Save results to CSV
        for config in motif_configs:
            if config.name in aligned_paths:
                design_df[f"aligned_{config.name}_path"] = aligned_paths[config.name]

        output_csv = BASE_DIR / f"design_input_sasa_filtered_aligned.csv"
        design_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}", flush=True)
        print(f"{'='*60}")
        print(f"COMPLETED: {BASE_DIR}")
        print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
