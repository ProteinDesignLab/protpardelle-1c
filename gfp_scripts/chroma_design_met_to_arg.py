from collections import defaultdict
from pathlib import Path

from chroma import Protein, Chroma
from chroma.constants.sequence import PROTEIN_TOKENS
import pandas as pd
import torch
import typer


# Amino acid to index mapping
AA_TO_IDX = {aa: i for i, aa in enumerate(PROTEIN_TOKENS)}


def index_to_seq(idx: torch.Tensor):
    return ''.join([PROTEIN_TOKENS[i] for i in idx])


def design(csv_fp: Path, rm_aa: str = 'C', num_samples: int = 8):
    """
    Wrapper around chromadesign, requires input csv with columns "pdb_path", "fix_pos"

    Modifications from original:
    - Correctly parses comma-separated fix_pos
    - Overrides MET to ARG at the MET position in fix_pos
    - Always bans CYS from design (in addition to any other banned residues)

    Note: fix_pos is 1-indexed, matching PDB 1-indexing
    """
    # Always ban CYS, plus any additional residues specified
    banned_residues = set(rm_aa)
    banned_residues.add('C')  # Always ban cysteine
    banned_residues = list(banned_residues)

    chroma = Chroma()

    df = pd.read_csv(csv_fp)

    df_out = defaultdict(list)

    for df_idx, (pdb_path, fix_pos) in enumerate(zip(df['pdb_path'], df['fix_pos'])):
        protein = Protein(pdb_path, device='cuda')
        protein.canonicalize()
        X, C, S = protein.to_XCS()

        # Parse fix_pos (comma-separated)
        fix_pos_list = [int(x) for x in str(fix_pos).split(',')]

        # Find MET position - it's one of the last two positions in fix_pos
        # The last two are the discontiguous MET and GLU positions
        met_pos_pdb = fix_pos_list[-2]  # Second to last is MET

        # Convert PDB 1-indexed to 0-indexed for tensor
        met_pos_tensor = met_pos_pdb - 1

        # Override MET to ARG in the sequence
        arg_idx = AA_TO_IDX['R']
        S_modified = S.clone()
        S_modified[0, met_pos_tensor] = arg_idx

        print(f"\n[{df_idx}] Processing: {Path(pdb_path).stem}")
        print(f"  fix_pos: {fix_pos}")
        print(f"  MET position (PDB): {met_pos_pdb}, tensor idx: {met_pos_tensor}")
        print(f"  Original residue at MET pos: {PROTEIN_TOKENS[S[0, met_pos_tensor].item()]}")
        print(f"  Changed to: {PROTEIN_TOKENS[S_modified[0, met_pos_tensor].item()]}")

        X = torch.cat([X] * num_samples, dim=0)
        C = torch.cat([C] * num_samples, dim=0)
        S = torch.cat([S_modified] * num_samples, dim=0)

        # Build mask for positions to design (everything except fix_pos)
        mask_sample = None
        if fix_pos != '' and not str(fix_pos) == 'nan':
            design_selection = 'not resid ' + '+'.join([str(idx) for idx in fix_pos_list])
            if isinstance(design_selection, str):
                design_selection = protein.get_mask(design_selection)
            mask_sample = design_selection

        X_sample, S_sample, _, scores = chroma.design_network.sample(
            X,
            C,
            S,
            t=0.5,
            mask_sample=mask_sample,
            temperature_S=0.01,
            temperature_chi=1e-3,
            ban_S=banned_residues,
            sampling_method="potts",
            regularization="LCP",
            potts_sweeps=500,
            potts_proposal="dlmc",
            verbose=False,
            symmetry_order=None,
            return_scores=True,
        )

        repack_save_dir = csv_fp.parent / f"{csv_fp.stem}_chroma_repack"
        repack_save_dir.mkdir(exist_ok=True)

        for bi in range(num_samples):
            protein.sys.update_with_XCS(X_sample[bi:bi+1], C=None, S=S_sample[bi:bi+1])

            # Short filename: d{row_index}_{sample_index}.pdb
            out_filename = f"d{df_idx}_{bi}.pdb"
            save_fp = str((repack_save_dir / out_filename).resolve())

            protein.to(save_fp)

            seq = index_to_seq(S_sample[bi])
            print(f"  Design {bi}: {seq}")
            print(f"    Residue at MET pos: {seq[met_pos_tensor]}")
            print(f"    Energy: {scores['U_sample'][bi].item():.2f}")

            # Copy all input columns first
            for colname in df.columns:
                df_out[colname].append(df[colname][df_idx])

            # Add design output columns
            df_out['design_idx'].append(bi)
            df_out['design_pdb'].append(out_filename)
            df_out['design_pdb_path'].append(save_fp)
            df_out['seq'].append(seq.replace(':', ''))
            df_out['energy'].append(scores['U_sample'][bi].item())
            df_out['met_pos_pdb'].append(met_pos_pdb)
            df_out['residue_at_met_pos'].append(seq[met_pos_tensor])

    df_out = pd.DataFrame(df_out)
    df_out_fp = csv_fp.parent / f"{csv_fp.stem}_design_output.csv"
    df_out.to_csv(df_out_fp, index=False)
    print(f"\nSaved output to {df_out_fp}")


if __name__ == '__main__':
    typer.run(design)
