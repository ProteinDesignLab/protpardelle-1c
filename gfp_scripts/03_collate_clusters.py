"""
Parse foldseek cluster output and generate design input CSV with fix_pos.

The fix_pos is determined by parsing each PDB structure to find:
1. A contiguous stretch of mostly non-GLY residues (the chromophore core, may include
   some GLY within, including the TYG chromophore)
2. MET residue outside the stretch
3. GLU residue outside the stretch

The fix_pos includes all indices from min to max of the stretch (including any GLY
within) plus the MET and GLU indices.
"""

import re
from pathlib import Path

import pandas as pd


def extract_dir_from_sample_name(sample_name: str) -> str:
    """Extract the directory name from a sample name.

    Pattern: ...-sample_<dirname>_<number>
    """
    match = re.search(r'-sample_(.+)_\d+$', sample_name)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract directory from sample name: {sample_name}")


def get_pdb_path(sample_name: str, base_dir: Path) -> Path:
    """Get the full PDB path for a sample name."""
    dir_name = extract_dir_from_sample_name(sample_name)
    return base_dir / dir_name / f"{sample_name}.pdb"


def parse_residues_from_pdb(pdb_path: Path) -> dict:
    """Parse residue types from a PDB file.

    Returns:
        Dict mapping residue index (int) to residue type (str, 3-letter code)
    """
    residues = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # PDB format: columns 18-20 are residue name, 23-26 are residue number
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                if res_num not in residues:
                    residues[res_num] = res_name
    return residues


def compute_fix_pos(pdb_path: Path) -> str:
    """Compute fix_pos indices for a PDB structure.

    The chromophore region consists of:
    1. A long stretch of residues (may include some GLY)
    2. A MET residue outside the stretch
    3. A GLU residue outside the stretch

    Returns:
        Comma-separated string of 0-indexed fix_pos indices
    """
    residues = parse_residues_from_pdb(pdb_path)

    # Find all residue indices
    all_indices = sorted(residues.keys())

    # Find non-GLY residues
    non_gly = {idx: res for idx, res in residues.items() if res != 'GLY'}

    # Find MET and GLU indices (should be isolated from the main stretch)
    met_indices = [idx for idx, res in non_gly.items() if res == 'MET']
    glu_indices = [idx for idx, res in non_gly.items() if res == 'GLU']

    # The stretch residues are all non-GLY except MET and GLU
    stretch_candidates = [idx for idx, res in non_gly.items()
                          if res not in ('MET', 'GLU')]

    if not stretch_candidates:
        # Fallback: just use non-GLY indices
        fix_pos = sorted(non_gly.keys())
    else:
        # Find the bounds of the stretch
        stretch_start = min(stretch_candidates)
        stretch_end = max(stretch_candidates)

        # Include all indices from start to end (including GLY within)
        stretch_indices = list(range(stretch_start, stretch_end + 1))

        # Also include MET and GLU that are closest to the stretch
        # (there should be exactly one of each in the motif)
        fix_pos = sorted(set(stretch_indices + met_indices + glu_indices))

    # Convert to 0-indexed (PDB uses 1-indexed residue numbers)
    # But we need to map to the actual position in the sequence (0-indexed)
    # Since the structure may not start at residue 1, we need to be careful

    # Create a mapping from residue index to 0-indexed position
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    # Convert fix_pos to 0-indexed positions
    fix_pos_0indexed = [idx_to_pos[idx] for idx in fix_pos if idx in idx_to_pos]

    return ','.join(map(str, sorted(fix_pos_0indexed)))


def process_cluster_file(cluster_tsv: Path, samples_dir: Path, save_dir: Path):
    """Process a single cluster file and save results."""
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {cluster_tsv}")
    print(f"Output dir: {save_dir}")
    print(f"{'='*60}")

    # Read cluster file (TSV with cluster_rep, cluster_member columns)
    df = pd.read_csv(cluster_tsv, sep='\t', header=None, names=['cluster_rep', 'cluster_member'])

    # Get unique cluster representatives
    cluster_reps = df['cluster_rep'].unique()
    print(f"Found {len(cluster_reps)} unique cluster representatives")

    # Process each cluster rep
    results = []
    for i, rep in enumerate(cluster_reps):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(cluster_reps)}...")

        try:
            pdb_path = get_pdb_path(rep, samples_dir)
            if not pdb_path.exists():
                print(f"Warning: PDB not found: {pdb_path}")
                continue

            fix_pos = compute_fix_pos(pdb_path)

            results.append({
                'cluster_rep': rep,
                'pdb_path': str(pdb_path),
                'fix_pos': fix_pos,
            })
        except Exception as e:
            print(f"Error processing {rep}: {e}")
            continue

    # Create output dataframe
    df_out = pd.DataFrame(results)

    # Add cluster size information
    cluster_sizes = df.groupby('cluster_rep').size().reset_index(name='cluster_size')
    df_out = df_out.merge(cluster_sizes, on='cluster_rep', how='left')

    # Sort by cluster size (descending)
    df_out = df_out.sort_values('cluster_size', ascending=False)

    # Save
    output_path = save_dir / "design_input.csv"
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} entries to {output_path}")


def main():
    samples_dir = Path("/scratch/users/tianyulu/gfp/denovo/samples_flat")

    # Define input/output pairs
    cluster_configs = [
        {
            'cluster_tsv': Path("/scratch/users/tianyulu/gfp/denovo/foldseek_05_cluster.tsv"),
            'save_dir': Path("/scratch/users/tianyulu/gfp/denovo/chroma_05"),
        },
        {
            'cluster_tsv': Path("/scratch/users/tianyulu/gfp/denovo/foldseek_08_cluster.tsv"),
            'save_dir': Path("/scratch/users/tianyulu/gfp/denovo/chroma_08"),
        },
    ]

    for config in cluster_configs:
        process_cluster_file(
            cluster_tsv=config['cluster_tsv'],
            samples_dir=samples_dir,
            save_dir=config['save_dir'],
        )


if __name__ == "__main__":
    main()
