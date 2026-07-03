from collections import defaultdict
from pathlib import Path

from chroma import Protein, Chroma
from chroma.constants.sequence import PROTEIN_TOKENS
import pandas as pd
import torch
import typer


def index_to_seq(idx: torch.Tensor):
    return ''.join([PROTEIN_TOKENS[i] for i in idx])


def design(csv_fp: Path, rm_aa: str = 'C', num_samples: int = 32, t: float = 0.0):
    """
    Wrapper around chroma design for array job processing.
    Input CSV should have columns: "pdb_path", "fix_pos", "pdb_stem"

    #! Note: fix_pos is 1-indexed, matching PDB 1-indexing
    """
    chroma = Chroma()

    df = pd.read_csv(csv_fp)

    df_out = defaultdict(list)

    for df_idx, row in df.iterrows():
        pdb_path = row['pdb_path']
        fix_pos = row['fix_pos'] if 'fix_pos' in row else ''
        pdb_stem = row['pdb_stem'] if 'pdb_stem' in row else Path(pdb_path).stem

        print(f"Processing {df_idx + 1}/{len(df)}: {pdb_stem}")

        try:
            protein = Protein(pdb_path, device='cuda')
            protein.canonicalize()
            X, C, S = protein.to_XCS()

            X = torch.cat([X] * num_samples, dim=0)
            C = torch.cat([C] * num_samples, dim=0)
            S = torch.cat([S] * num_samples, dim=0)

            mask_sample = None
            if pd.notna(fix_pos) and str(fix_pos).strip() != '':
                design_selection = 'not resid ' + '+'.join([str(idx) for idx in str(fix_pos).split()])
                if isinstance(design_selection, str):
                    design_selection = protein.get_mask(design_selection)
                mask_sample = design_selection

            X_sample, S_sample, _, scores = chroma.design_network.sample(
                X,
                C,
                S,
                t=t,
                mask_sample=mask_sample,
                temperature_S=0.01,
                temperature_chi=1e-3,
                ban_S=list(rm_aa),
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

                save_fp = str((repack_save_dir / f"{pdb_stem}_chroma_remove_{rm_aa}_design_{bi}.pdb").resolve())
                protein.to(save_fp)

                df_out['input_pdb_stem'].append(pdb_stem)
                df_out['input_pdb_path'].append(pdb_path)
                df_out['fix_pos'].append(fix_pos)
                df_out['t'].append(t)
                df_out['design_idx'].append(bi)
                seq = index_to_seq(S_sample[bi])
                df_out['seq'].append(seq.replace(':', ''))
                df_out['seq_len'].append(len(seq.replace(':', '')))
                df_out['energy'].append(scores['U_sample'][bi].item())
                df_out['repack_pdb_path'].append(save_fp)

        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            # Record the error for this structure
            for bi in range(num_samples):
                df_out['input_pdb_stem'].append(pdb_stem)
                df_out['input_pdb_path'].append(pdb_path)
                df_out['fix_pos'].append(fix_pos)
                df_out['t'].append(t)
                df_out['design_idx'].append(bi)
                df_out['seq'].append('')
                df_out['seq_len'].append(0)
                df_out['energy'].append(float('nan'))
                df_out['repack_pdb_path'].append('')
            continue

    df_out = pd.DataFrame(df_out)
    df_out_fp = csv_fp.parent / f"{csv_fp.stem}_design_output.csv"
    df_out.to_csv(df_out_fp, index=False)
    print(f"Saved output to {df_out_fp}")


if __name__ == '__main__':
    typer.run(design)
