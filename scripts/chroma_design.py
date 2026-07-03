from collections import defaultdict
from pathlib import Path

from chroma import Protein, Chroma
from chroma.constants.sequence import PROTEIN_TOKENS
import pandas as pd
import torch
import typer


def index_to_seq(idx: torch.Tensor):
    return ''.join([PROTEIN_TOKENS[i] for i in idx])


def design(csv_fp: Path, rm_aa: str = 'C', num_samples: int = 8):
    """
    Wrapper around chromadesign, requires input csv with columns "pdb_path", "fix_pos"

    #! Note: fix_pos is 1-indexed, matching PDB 1-indexing
    """
    chroma = Chroma()

    df = pd.read_csv(csv_fp)

    df_out = defaultdict(list)

    for df_idx, (pdb_path, fix_pos) in enumerate(zip(df['pdb_path'], df['fix_pos'])):
        protein = Protein(pdb_path, device='cuda')
        protein.canonicalize()
        X, C, S = protein.to_XCS()

        X = torch.cat([X] * num_samples, dim=0)
        C = torch.cat([C] * num_samples, dim=0)
        S = torch.cat([S] * num_samples, dim=0)

        # design_selection = 'not chain A and resid ' + '+'.join([str(idx) for idx in fix_pos.split()])
        mask_sample = None
        if fix_pos != '' and not str(fix_pos) == 'nan':
            design_selection = 'not resid ' + '+'.join([str(idx) for idx in fix_pos.split()])
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

            unique_pdb_name = f"{Path(pdb_path).parent.parent.name}-{Path(pdb_path).parent.name}-{Path(pdb_path).stem}"
            save_fp = str((repack_save_dir / f"{unique_pdb_name}_chroma_remove_{rm_aa}_design_{bi}.pdb").resolve())

            protein.to(save_fp)

            df_out['pdb_path'].append(pdb_path)
            df_out['fix_pos'].append(fix_pos)
            if 'orig_fix_pos' in df.columns:
                df_out['orig_fix_pos'].append(df['orig_fix_pos'][df_idx])
            seq = index_to_seq(S_sample[bi])
            # if C[bi].sum() > 0:
            #     seq_chains = []
            #     ci_start = 0
            #     change_idx = ((C[bi, 1:] != C[bi, :-1]).nonzero().flatten() + 1)
            #     for ci in change_idx:
            #         seq_chains.append(seq[ci_start : ci])
            #         ci_start = ci
            #     seq = ':'.join(seq_chains)
            print(seq)
            print("Energy: ", scores['U_sample'][bi].item())
            df_out['seq'].append(seq.replace(':', ''))
            df_out['energy'].append(scores['U_sample'][bi].item())
            df_out['repack_pdb_path'].append(save_fp)
            for colname in df.columns:
                df_out[f"design_input_{colname}"].append(df[colname][df_idx])

    df_out = pd.DataFrame(df_out)
    df_out_fp = csv_fp.parent / f"{csv_fp.stem}_design_output.csv"
    df_out.to_csv(df_out_fp, index=False)
    print(f"Saved output to {df_out_fp}")


if __name__ == '__main__':
    typer.run(design)
