"""
Replace "      A    " with "           " in ATOM lines of pdb files in specified directories,
then sort the pdb files using pdb_sort and renumber atoms to be strictly increasing using pdb_reatom.
"""

from pathlib import Path
import os


data_dir = [
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_fixed/chroma/pre_cyclic_superimposed"),
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_fixed/chroma/post_cyclic_superimposed"),
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_flex/chroma/pre_cyclic_superimposed"),
    Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_flex/chroma/post_cyclic_superimposed"),
]

for dd in data_dir:
    for fp in dd.glob("*.pdb"):
        with open(fp, "r") as f:
            clean_lines = []
            lines = f.readlines()
            for line in lines:
                if line.startswith("ATOM"):
                    line = line.replace("      A    ", "           ")
                    clean_lines.append(line)
        with open(fp, "w") as f:
            f.writelines(clean_lines)

        os.system(f"pdb_sort -R {fp} | pdb_reatom -1 > {fp.parent / fp.stem}_sorted.pdb")
