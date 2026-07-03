from collections import defaultdict
from pathlib import Path
import subprocess

import pandas as pd
from tqdm import tqdm


design_fps = [
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_flex/chroma/all_design_output_with_sasa_lt_100_aligned_chromophore.csv"),
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_fixed/chroma/all_design_output_with_sasa_lt_100_aligned_chromophore.csv"),
]


def fixpos_to_contig(pdb_length, fixpos_ints):
    """
    Convert fixed positions into a motif contig string.
    
    Args:
        pdb_length: Total length of the PDB structure
        fixpos_ints: List of fixed position integers (1-indexed)
    
    Returns:
        Contig string in the format: length;Apos-pos;length;...
    """
    if not fixpos_ints:
        return f"0-{pdb_length - 1}"
    
    # Sort fixed positions
    fixpos_ints = sorted(fixpos_ints)
    
    contig_parts = []
    current_pos = 0
    
    i = 0
    while i < len(fixpos_ints):
        # Add gap before fixed position(s)
        gap_length = fixpos_ints[i] - 1 - current_pos
        if gap_length > 0:
            contig_parts.append(f"{gap_length}-{gap_length}")
        
        # Find consecutive fixed positions
        start_fix = fixpos_ints[i]
        end_fix = fixpos_ints[i]
        
        while i + 1 < len(fixpos_ints) and fixpos_ints[i + 1] == fixpos_ints[i] + 1:
            i += 1
            end_fix = fixpos_ints[i]
        
        # Add fixed position(s)
        contig_parts.append(f"A{start_fix}-{end_fix}")
        current_pos = end_fix
        i += 1
    
    # Add remaining positions after last fixed position
    remaining = pdb_length - current_pos
    if remaining > 0:
        contig_parts.append(f"{remaining}-{remaining}")
    
    return ";".join(contig_parts)


for design_fp in design_fps:
    df = pd.read_csv(design_fp)

    # rename some columns from chroma
    cols_to_rename = ["seq", "energy", "repack_pdb_path"]
    df.rename(columns={col: f"chroma_{col}" for col in cols_to_rename}, inplace=True)

    df_caliby = defaultdict(list)

    for curr_seq, pre_pdb_fp, post_pdb_fp, fix_pos in tqdm(zip(df["chroma_seq"], df["aligned_pre_cyclic_path"], df["aligned_post_cyclic_path"], df["fix_pos"]), total=len(df)):
        pre_pdb_fp = Path(pre_pdb_fp)
        pre_pdb_fp = pre_pdb_fp.parent / "sorted" / f"{pre_pdb_fp.stem}_sorted.pdb"
        df_caliby["pre_pdb_path"].append(str(pre_pdb_fp))

        post_pdb_fp = Path(post_pdb_fp)
        post_pdb_fp = post_pdb_fp.parent / "sorted" / f"{post_pdb_fp.stem}_sorted.pdb"
        df_caliby["post_pdb_path"].append(str(post_pdb_fp))

        cmd = f"grep '^ATOM' {str(pre_pdb_fp)} | awk '{{print $5 $6}}' | uniq | wc -l"
        pdb_length = int(subprocess.check_output(cmd, shell=True).strip())
        df_caliby["pdb_length"].append(pdb_length)

        fixpos_list = fix_pos.split(" ")
        caliby_fixpos = ",".join(f"A{i}" for i in fixpos_list)

        # fill in fixed_pos_override_seq at the second last fixed pos (always ARG, not MET)
        arg_pos = fixpos_list[-2]

        # remove the second last fixed pos from fixed_pos_scn if that position is MET (since ARG does not exist in the structure)
        if curr_seq[int(arg_pos) - 1] == "M":
            df_caliby["fixed_pos_override_seq"].append(f"A{arg_pos}:R")
            caliby_fixpos_scn = ",".join(f"A{i}" for i in fixpos_list if i != arg_pos)
            df_caliby["fixed_pos_scn"].append(caliby_fixpos_scn)
        else:
            df_caliby["fixed_pos_override_seq"].append("")
            df_caliby["fixed_pos_scn"].append(caliby_fixpos)

        fixpos_ints = [int(x) for x in fixpos_list]
        contig = fixpos_to_contig(pdb_length, fixpos_ints)
        df_caliby["motif_contig_override"].append(contig)  # used for fixed motif ensemble generation

        df_caliby["fixed_pos_seq"].append(caliby_fixpos)

    for k, v in df_caliby.items():
        df[k] = v

    df_pre = df.copy()
    df_pre["pdb_path"] = df_pre["pre_pdb_path"]
    df_post = df.copy()
    df_post["pdb_path"] = df_post["post_pdb_path"]

    df_pre.to_csv(design_fp.parent / f"{design_fp.stem}_caliby_inputs_pre.csv", index=False)
    df_post.to_csv(design_fp.parent / f"{design_fp.stem}_caliby_inputs_post.csv", index=False)

    # also save a version without motif sidechain conditioning
    df_pre_nosc = df_pre.copy()
    df_pre_nosc["fixed_pos_scn"] = [""] * len(df_pre_nosc)
    df_pre_nosc.to_csv(design_fp.parent / f"{design_fp.stem}_caliby_inputs_pre_nosc.csv", index=False)
    df_post_nosc = df_post.copy()
    df_post_nosc["fixed_pos_scn"] = [""] * len(df_post_nosc)
    df_post_nosc.to_csv(design_fp.parent / f"{design_fp.stem}_caliby_inputs_post_nosc.csv", index=False)

    print(f"Saved caliby input csvs to {design_fp.parent}")
