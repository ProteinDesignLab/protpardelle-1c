"""
Compute SASA for all structures using PULCHRA with TYR packing.
Saves results to design_input_with_sasa.csv for each cluster config.
"""

import subprocess
import tempfile
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool, cpu_count

from Bio.PDB import PDBParser, PDBIO, ShrakeRupley
import pandas as pd
from tqdm import tqdm

PULCHRA_PATH = "/oak/stanford/groups/possu/tianyu/pulchra_306/pulchra"


def compute_sasa_with_pulchra(args):
    """Worker function for parallel SASA computation with PULCHRA TYR packing."""
    pdb_path, fix_pos_str = args

    try:
        fix_pos = [int(p) for p in fix_pos_str.split(',')]
        common_pos = fix_pos[:7] + fix_pos[11:]
        pdb_residue_nums = set(p + 1 for p in common_pos)

        parser = PDBParser(QUIET=True)
        io = PDBIO()

        # Load original structure and save motif residue coordinates
        structure_orig = parser.get_structure('orig', pdb_path)
        motif_residues_orig = {}
        for chain in structure_orig[0]:
            for res in chain:
                if res.id[1] in pdb_residue_nums:
                    motif_residues_orig[res.id[1]] = deepcopy(res)

        # Load structure and mutate GLY to TYR outside motif
        structure = parser.get_structure('', pdb_path)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[1] not in pdb_residue_nums and residue.resname == 'GLY':
                        residue.resname = 'TYR'

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdb = Path(tmpdir) / "input.pdb"
            io.set_structure(structure)
            io.save(str(tmp_pdb))

            # Run PULCHRA
            subprocess.run(
                [PULCHRA_PATH, str(tmp_pdb)],
                capture_output=True, cwd=tmpdir
            )

            rebuilt_pdb = Path(tmpdir) / "input.rebuilt.pdb"
            if rebuilt_pdb.exists():
                rebuilt_structure = parser.get_structure('', rebuilt_pdb)
                first_chain = list(rebuilt_structure[0].get_chains())[0]

                # Replace motif residues with originals
                for res_id, orig_res in motif_residues_orig.items():
                    if res_id in [r.id[1] for r in first_chain]:
                        rebuilt_res = first_chain[res_id]
                        first_chain.detach_child(rebuilt_res.id)
                        orig_res_copy = deepcopy(orig_res)
                        orig_res_copy.id = (' ', res_id, ' ')
                        first_chain.add(orig_res_copy)

                sr = ShrakeRupley()
                sr.compute(rebuilt_structure, level='R')

                sasa = sum(
                    first_chain[ri].sasa
                    for ri in pdb_residue_nums
                    if ri in [r.id[1] for r in first_chain]
                )
                return sasa

        return float('inf')

    except Exception as e:
        return float('inf')


def process_cluster_file(design_input_csv: Path, output_dir: Path):
    """Process a single cluster file and compute SASA for all structures."""
    print(f"\n{'='*60}")
    print(f"Processing: {design_input_csv}")
    print(f"{'='*60}", flush=True)

    if not design_input_csv.exists():
        print(f"Warning: {design_input_csv} not found, skipping...")
        return

    output_csv = output_dir / "design_input_with_sasa.csv"

    # Check if already computed
    if output_csv.exists():
        print(f"Output already exists: {output_csv}")
        print("Delete it to recompute.", flush=True)
        return

    # Load design input
    df = pd.read_csv(design_input_csv)
    print(f"Loaded {len(df)} structures", flush=True)

    # Prepare arguments for parallel computation
    args = [(row['pdb_path'], row['fix_pos']) for _, row in df.iterrows()]

    # Parallel SASA computation
    n_workers = min(cpu_count(), 20)
    print(f"Computing SASA with {n_workers} workers (TYR packing)...", flush=True)

    with Pool(n_workers) as pool:
        all_sasa = list(tqdm(
            pool.imap(compute_sasa_with_pulchra, args),
            total=len(args)
        ))

    df['sasa'] = all_sasa
    df.to_csv(output_csv, index=False)
    print(f"Saved all results to: {output_csv}", flush=True)

    # Print summary statistics
    valid_sasa = [s for s in all_sasa if s != float('inf')]
    if valid_sasa:
        print(f"\nSASA Statistics:")
        print(f"  Total: {len(all_sasa)}")
        print(f"  Valid: {len(valid_sasa)}")
        print(f"  Min: {min(valid_sasa):.2f}")
        print(f"  Max: {max(valid_sasa):.2f}")
        print(f"  Mean: {sum(valid_sasa)/len(valid_sasa):.2f}")
        print(f"  < 50: {sum(1 for s in valid_sasa if s < 50)}")
        print(f"  < 100: {sum(1 for s in valid_sasa if s < 100)}")
        print(f"  < 200: {sum(1 for s in valid_sasa if s < 200)}", flush=True)

    # Save filtered subset with SASA threshold 50
    SASA_THRESHOLD = 50
    df_filtered = df[df['sasa'] <= SASA_THRESHOLD].copy()
    filtered_csv = output_dir / f"design_input_sasa_lt_{SASA_THRESHOLD}.csv"
    df_filtered.to_csv(filtered_csv, index=False)
    print(f"\nSaved {len(df_filtered)} structures with SASA <= {SASA_THRESHOLD} to: {filtered_csv}", flush=True)


def main():
    # Input configurations
    configs = [
        (
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_05/design_input.csv"),
            Path("/scratch/users/tianyulu/gfp/denovo/chroma_05"),
        ),
    ]

    for design_input_csv, output_dir in configs:
        process_cluster_file(design_input_csv, output_dir)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
