"""
After generating pre and post cyclized ensembles, make pdb_to_conformers json for Caliby,
and add the "pdb_key" column to the design csv files.
"""

from collections import defaultdict
from pathlib import Path
import json

from natsort import natsorted
import numpy as np
import pandas as pd


base_data_dir = Path("/scratch/users/tianyulu/allatom-design-sampling-results/conformers/10302025-de_novo_gfp")
subdir_name = "protpardelle-1c/cc58-epoch416-sampling_sidechain_conditional_partial_diffusion-ss1.0-schurn0-ccstart0.0-dx0.0-dy0.0-dz0.0-rewind150"



for flex_or_flat in ["fixed", "flex"]:  # "flex" or "fixed"
    pdb_to_conformers = defaultdict(dict)

    for conformer_dir in (base_data_dir / f"{flex_or_flat}_pre_cyclic" / subdir_name).iterdir():
        if not conformer_dir.is_dir():
            continue

        post_cyclic_dir = base_data_dir / f"{flex_or_flat}_post_cyclic" / subdir_name / conformer_dir.name

        pre_cyclic_paths = natsorted(list(conformer_dir.glob("*.pdb")))
        post_cyclic_paths = natsorted(list(post_cyclic_dir.glob("*.pdb")))

        primary_conformer_path = pre_cyclic_paths[0]
        assert not primary_conformer_path.stem.startswith("sample")

        pre_cyclic_paths = pre_cyclic_paths[1:] # exclude the primary conformer
        post_cyclic_paths = post_cyclic_paths[1:] # exclude the primary conformer

        # fill in the pdb to conformers dict with paths and weights
        conformer_paths = np.array(pre_cyclic_paths + post_cyclic_paths)
        np.random.shuffle(conformer_paths)
        conformer_paths = [primary_conformer_path] + list(conformer_paths)
        pdb_weights = [1.0] * len(conformer_paths)

        pdb_to_conformers[primary_conformer_path.stem] = [(str(p), w) for p, w in zip(conformer_paths, pdb_weights)]
        
        assert conformer_dir.name == primary_conformer_path.stem

    design_csv_data_dir = Path(f"/scratch/users/tianyulu/protpardelle-1c-sampling-results/10172025-de_novo_gfp_{flex_or_flat}/chroma")

    design_csv_fps = list(design_csv_data_dir.glob("all_design_output_with_sasa_lt_100_aligned_chromophore_caliby_inputs_pre*.csv"))

    for design_csv_fp in design_csv_fps:
        design_df = pd.read_csv(design_csv_fp)

        design_df['pdb_key'] = [Path(pdb_path).stem for pdb_path in design_df['pdb_path']]
        design_df.to_csv(design_csv_fp.parent / f"{design_csv_fp.stem}_with_pdb_key.csv", index=False)

    with open(base_data_dir / f"{flex_or_flat}_pdb_to_conformers.json", "w") as f:
        json.dump(pdb_to_conformers, f, indent=4)
