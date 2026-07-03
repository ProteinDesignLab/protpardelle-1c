"""
Given Caliby seq_des_outputs.csv files, generate AlphaFold2 input files
evenly distributing the number of designs to predict
"""

from pathlib import Path

import pandas as pd


af2_batch_size = 100


result_root_dir = Path("/scratch/users/tianyulu/allatom-design-sampling-results")

data_dirs = [
    # result_root_dir / "gfp_ensemble_design",
    # result_root_dir / "pre_cyclic_only_gfp_ensemble_design",
    result_root_dir / "post_cyclic_only_gfp_ensemble_design",
]


for data_dir in data_dirs:

    all_dfs = []

    for subdir in data_dir.iterdir():
        seq_des_csv_fp = list(subdir.glob("*/seq_des_outputs.csv"))
        if len(seq_des_csv_fp) == 1:

            seq_des_csv_fp = seq_des_csv_fp[0]

            df = pd.read_csv(seq_des_csv_fp)
            df["group"] = subdir.name
            
            # take the lowest energy sequence within each example_id
            df = df.loc[df.groupby("example_id")["U"].idxmin()]

            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)


    num_designs = len(full_df)

    # save af2 input files, each having at most af2_batch_size designs
    out_dir = data_dir / "af2"
    out_dir.mkdir(exist_ok=True)

    for i in range(0, num_designs, af2_batch_size):
        batch_df = full_df.iloc[i : i + af2_batch_size]
        batch_fp = out_dir / f"seq_des_outputs_part_{i//af2_batch_size}.csv"
        batch_df.to_csv(batch_fp, index=False)
