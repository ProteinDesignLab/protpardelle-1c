from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import pandas as pd
import torch
from jax.tree_util import tree_map
from scipy.special import softmax

from pathlib import Path
from tqdm import tqdm
import typer


alphabet = "ARNDCQEGHILKMFPSTWYVX"


def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0, :, 1]

    bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    sm_contacts = softmax(output["distogram_logits"], -1)[0]
    sm_contacts = sm_contacts[..., bins < 8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    o = {
        "pae": pae[mask, :][:, mask],
        "plddt": plddt[mask],
        "sm_contacts": sm_contacts[mask, :][:, mask],
        "xyz": xyz[mask],
    }
    return o


def run_esmfold(
    seq_des_csv: Path,
    model_path: Path = Path("/oak/stanford/groups/possu/tianyu/models/esmfold.model"),
    num_recycles: int = 3,
    chain_linker: int = 25,
    save_every: int = 1000,
):
    """
    Given output of caliby, predict structures with ESMFold
    """
    df = pd.read_csv(seq_des_csv)
    groups = df["group"].to_list()
    example_ids = df["example_id"].to_list()
    U = df["U"].to_list()
    seqs = df["seq"].to_list()

    data = []

    # Load model once
    print("Loading ESMFold model...")
    model = torch.load(str(model_path), weights_only=False)
    model.eval().cuda().requires_grad_(False)

    for n, (group_name, example_id, u_val, seq) in tqdm(
        enumerate(zip(groups, example_ids, U, seqs)), total=len(groups)
    ):
        # output directory name based on group name
        out_dir = seq_des_csv.parent / "esmfold_pred" / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_fp = out_dir / f"{example_id}.pdb"

        # Clean sequence
        sequence = re.sub("[^A-Z:]", "", str(seq).replace("/", ":").upper())
        sequence = re.sub(":+", ":", sequence)
        sequence = re.sub("^[:]+", "", sequence)
        sequence = re.sub("[:]+$", "", sequence)

        length = len(sequence.replace(":", ""))

        # Optimize chunk size based on length
        if length > 700:
            model.set_chunk_size(64)
        else:
            model.set_chunk_size(128)

        # Run ESMFold
        torch.cuda.empty_cache()
        output = model.infer(
            sequence,
            num_recycles=num_recycles,
            chain_linker="X" * chain_linker,
            residue_index_offset=512,
        )

        # Get PDB and metrics
        pdb_str = model.output_to_pdb(output)[0]
        output = tree_map(lambda x: x.cpu().numpy(), output)

        ptm = output["ptm"][0]
        plddt = output["plddt"][0, ..., 1].mean()
        O = parse_output(output)
        pae_mean = O["pae"].mean()

        # Save PDB file
        with open(output_fp, "w") as out:
            out.write(pdb_str)

        # Store results
        labels = ["plddt", "ptm", "pae", "group", "example_id", "U", "seq", "esmfold_pred_pdb"]
        sublist = [float(plddt), float(ptm), float(pae_mean), group_name, example_id, u_val, seq, str(output_fp)]

        data.append(sublist)
        if (n + 1) % save_every == 0:
            df_results = pd.DataFrame(data, columns=labels)
            df_results.to_csv(out_dir / f"{seq_des_csv.stem}_esmfold_results_{n}.csv", index=False)

    df_results = pd.DataFrame(data, columns=labels)
    df_results.to_csv(out_dir / f"{seq_des_csv.stem}_esmfold_results_all.csv", index=False)


def runner(seq_des_csv_fp: Path):
    run_esmfold(
        seq_des_csv_fp,
        num_recycles=3,
        save_every=1000,
    )


if __name__ == "__main__":
    typer.run(runner)