# Script included for reference only, environment does not have Jax, colabdesign, PyRosetta support

"""Use AF2-Multimer to design interface residues and ProteinMPNN to design binder scaffold residues."""

import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyrosetta
import typer
from Bio.PDB import PDBParser, Selection
from colabdesign import clear_mem, mk_afdesign_model, mk_mpnn_model

dalphaball_path = Path(__file__).parent / "DAlphaBall.gcc"
pyrosetta.init(
    f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1"
)

from bindcraft_utils import (
    add_helix_loss,
    add_i_ptm_loss,
    add_rg_loss,
    calc_ss_percentage,
    pr_relax,
    score_interface,
)

parser = PDBParser(QUIET=True)


af2_param_dir = "/your/path/to/models/af2/params"
target_dir = Path("/your/path/to/farfalle/motifs/bindcraft")
target_stems = [
    "0_PD-1",
    "1_PD-L1",
    "2_IFNAR2",  # "2_IFNAR2",
    "3_CD45(d2)",
    "4_CD45(d3-4)",
    "5_Claudin1",
    "6_BBF-14(1)",
    "7_BBF-14(2)",
    "8_CrSAS-6(1)",
    "9_CrSAS-6(2)",
    "10_Derf7",
    "11_Derf21",
    "12_Betv1",
    "13_SpCas9",
    "14_CbAgo(N+PIWI)",
    "15_CbAgo(PAZ)",
    "16_HER2(AAV)",
    "17_PDL1(AAV)",
]
target_pdb_paths = [target_dir / f"{stem}.pdb" for stem in target_stems]
target_chains = [
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "A,B,C",
    "A,B,C,D",
    "A",
    "A,B",
    "A",
]
binder_chains = [
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "D",
    "E",
    "B",
    "C",
    "B",
]
design_models = [0, 1, 2, 3, 4]
prediction_models = [0, 1]
all_chains = [",".join([tc, bc]) for (tc, bc) in zip(target_chains, binder_chains)]


# run MPNN to generate sequences for binders
def mpnn_gen_sequence(
    pdb, mpnn_input_chains, interface_residues, T: float = 0.1, batch_size: int = 2
):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(
        backbone_noise=0.00, model_name="v_48_020", weights="soluble"
    )

    fixed_positions = "A," + interface_residues
    fixed_positions = fixed_positions.rstrip(
        ","
    )  # handle when interface residues is empty
    print(f"MPNN fixed positions: {fixed_positions}")

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(
        pdb_filename=str(pdb),
        chain=mpnn_input_chains,
        fix_pos=fixed_positions,
        rm_aa="C",
    )

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=T, num=1, batch=batch_size)

    return mpnn_sequences


def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]), 2)


# adapted from https://github.com/martinpacesa/BindCraft/blob/05702c435e2172a99c2b3faf87487badb6e54727/functions/colabdesign_utils.py#L94
def run_4stage_afdesign(
    input_pdb: Path,
    chain: str,
    chain_lengths: list,
    output_pdb: Path,
    logits_iterations: int = 50,
    soft_iterations: int = 75,
    temporary_iterations: int = 45,
    hard_iterations: int = 5,
    greedy_percentage: float = 1.0,
    greedy_iterations: int = 15,
    optimise_beta_extra_soft: int = 0,
    optimise_beta_extra_temp: int = 0,
    optimise_beta_recycles_design: int = 3,
    rm_aa: str = "C",
    seed: int = 1824,
):
    # soft_iterations: int = 75, temporary_iterations: int = 45, hard_iterations: int = 5

    binder_length = chain_lengths[-1]
    target_length = sum(chain_lengths[:-1])

    # option 1: fixbb protocol
    target_chain_str = ",".join(chain.split(",")[:-1])
    # af_model = mk_afdesign_model(protocol="fixbb", num_recycles=1, data_dir=af2_param_dir,
    #                             use_multimer=True, use_initial_guess=True, use_initial_atom_pos=False)
    # af_model.prep_inputs(pdb_filename=str(input_pdb), chain=chain,
    #                      rm_template=False, rm_template_seq=True, rm_template_sc=True, rm_template_ic=False, fix_pos=target_chain_str)

    # option 2: binder protocol
    # output sequence is only the binder chain sequence
    af_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=1,
        data_dir=af2_param_dir,
        use_multimer=True,
        use_initial_guess=True,
        use_initial_atom_pos=False,
    )
    af_model.prep_inputs(
        pdb_filename=str(input_pdb),
        target_chain=target_chain_str,
        binder_len=chain_lengths[-1],
        binder_chain=chain.split(",")[-1],
        rm_target_seq=False,
        rm_target_sc=False,
        use_binder_template=True,
        rm_binder_seq=True,
        rm_binder_sc=True,
        rm_template_ic=True,
        fix_pos=target_chain_str,
    )

    ### Update weights based on specified settings
    af_model.opt["weights"].update(
        {
            "pae": 0.4,
            "plddt": 0.1,
            "i_pae": 0.1,
            "con": 1.0,
            "i_con": 1.0,
        }
    )

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num": 2, "cutoff": 14.0, "binary": False, "seqsep": 9})
    af_model.opt["i_con"].update({"num": 2, "cutoff": 20.0, "binary": False})

    add_rg_loss(af_model, 0.3)
    add_i_ptm_loss(af_model, 0.05)
    add_helix_loss(af_model, -0.3)

    af_model._binder_len = binder_length
    af_model._target_len = target_length

    af_model.restart(rm_aa=rm_aa, seed=seed)

    # initial logits to prescreen trajectory
    print("Stage 1: Test Logits")
    af_model.design_logits(
        iters=logits_iterations,
        e_soft=0.9,
        models=design_models,
        num_models=1,
        sample_models=True,
        save_best=True,
    )

    # determine pLDDT of best iteration according to lowest 'loss' value
    initial_plddt = get_best_plddt(af_model, binder_length)

    # if best iteration has high enough confidence then continue
    if initial_plddt > 0.65:
        print("Initial trajectory pLDDT good, continuing: " + str(initial_plddt))
        # temporarily dump model to assess secondary structure
        af_model.save_pdb(output_pdb)
        _, beta, *_ = calc_ss_percentage(output_pdb, "B")
        os.remove(output_pdb)

        # if beta sheeted trajectory is detected then choose to optimise
        if float(beta) > 15:
            soft_iterations = soft_iterations + optimise_beta_extra_soft
            temporary_iterations = temporary_iterations + optimise_beta_extra_temp
            af_model.set_opt(num_recycles=optimise_beta_recycles_design)
            print("Beta sheeted trajectory detected, optimising settings")

        # how many logit iterations left
        logits_iter = soft_iterations - 50
        if logits_iter > 0:
            print("Stage 1: Additional Logits Optimisation")
            af_model.clear_best()
            af_model.design_logits(
                iters=logits_iter,
                e_soft=1,
                models=design_models,
                num_models=1,
                sample_models=True,
                ramp_recycles=False,
                save_best=True,
            )
            af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
            logit_plddt = get_best_plddt(af_model, binder_length)
            print("Optimised logit trajectory pLDDT: " + str(logit_plddt))
        else:
            logit_plddt = initial_plddt

        # perform softmax trajectory design
        print("Stage 2: Softmax Optimisation")
        af_model.clear_best()
        af_model.design_soft(
            temporary_iterations,
            e_temp=1e-2,
            models=design_models,
            num_models=1,
            sample_models=True,
            ramp_recycles=False,
            save_best=True,
        )
        softmax_plddt = get_best_plddt(af_model, binder_length)

        greedy_tries = math.ceil(binder_length * (greedy_percentage / 100))

        # perform one hot encoding
        if softmax_plddt > 0.65:
            print("Softmax trajectory pLDDT good, continuing: " + str(softmax_plddt))
            af_model.clear_best()
            print("Stage 3: One-hot Optimisation")
            af_model.design_hard(
                hard_iterations,
                temp=1e-2,
                models=design_models,
                num_models=1,
                sample_models=True,
                dropout=False,
                ramp_recycles=False,
                save_best=True,
            )
            onehot_plddt = get_best_plddt(af_model, binder_length)

            if onehot_plddt > 0.65:
                # perform greedy mutation optimisation
                print("One-hot trajectory pLDDT good, continuing: " + str(onehot_plddt))
                print("Stage 4: PSSM Semigreedy Optimisation")
                af_model.design_pssm_semigreedy(
                    soft_iters=0,
                    hard_iters=greedy_iterations,
                    tries=greedy_tries,
                    models=design_models,
                    num_models=1,
                    sample_models=True,
                    ramp_models=False,
                    save_best=True,
                )

            else:
                print(
                    "One-hot trajectory pLDDT too low to continue: " + str(onehot_plddt)
                )

        else:
            print("Softmax trajectory pLDDT too low to continue: " + str(softmax_plddt))

    else:
        print("Initial trajectory pLDDT too low to continue: " + str(initial_plddt))

    af_model.save_pdb(output_pdb)


def design(pdb_csv: Path):
    df = pd.read_csv(pdb_csv)
    pdb_paths = df["pdb_path"].to_list()

    df_out_fp = pdb_csv.parent / f"{pdb_csv.stem}_afdesign_mpnn_all.csv"

    df_progress_fp = pdb_csv.parent / f"{pdb_csv.stem}_afdesign_mpnn_progress.csv"

    # load from "checkpoint" if it exists
    df_out = defaultdict(list)
    if df_progress_fp.exists():
        df_progress = pd.read_csv(df_progress_fp)
        for col in df_progress.columns:
            df_out[col] = df_progress[col].to_list()

    for i, pdb_path in enumerate(pdb_paths):
        pdb_path = Path(pdb_path)

        clear_mem()

        target_path = None
        for si, stem in enumerate(target_stems):
            if stem in pdb_path.stem:
                target_idx = si
                target_path = str(target_pdb_paths[si].resolve())
                chain = all_chains[si]
                binder_chain_id = binder_chains[si]
                break
        if target_path is None:
            print(f"Target for {str(pdb_path)} not found!")
            raise AssertionError

        # AF2 Multimer design
        chain_lengths = []
        structure = parser.get_structure("s", pdb_path)
        for cc in Selection.unfold_entities(structure[0], "C"):
            chain_lengths.append(len(list(Selection.unfold_entities(cc, "R"))))

        af_design_dir = pdb_path.parent / "afdesign_100"
        af_design_dir.mkdir(exist_ok=True, parents=True)
        af_design_path = af_design_dir / f"{pdb_path.stem}_{i}_afdesign.pdb"

        # only run if it does not already exist in "checkpoint"
        if str(af_design_path) not in df_out["afdesign_pdb_path"]:
            run_4stage_afdesign(pdb_path, chain, chain_lengths, af_design_path)

            if af_design_path.exists():
                relax_dir = pdb_path.parent / "relax_100"
                relax_dir.mkdir(exist_ok=True, parents=True)
                relax_path = relax_dir / f"{pdb_path.stem}_{i}_relax.pdb"

                # Relax and get interface residues to fix during MPNN sequence design
                # afdesign relabels the target chain to A with 50 residue index gaps for missing regions
                # and binder chain is always B
                pr_relax(str(af_design_path), str(relax_path))

                interface_scores, interface_AA, interface_residues = score_interface(
                    str(relax_path), "B"
                )

                mpnn_aux = mpnn_gen_sequence(str(relax_path), "A,B", interface_residues)

                for si, mpnn_seq in enumerate(mpnn_aux["seq"]):
                    df_out["pdb_path"].append(str(relax_path))
                    df_out["afdesign_pdb_path"].append(str(af_design_path))

                    mpnn_seq = mpnn_seq.replace("/", ":")
                    df_out["seq"].append(mpnn_seq)

                    for k, v in interface_scores.items():
                        df_out[f"{k}"].append(v)
                    df_out[f"interface_AA"].append(interface_AA)
                    df_out[f"interface_residues"].append(interface_residues)

                df_progress = pd.DataFrame(df_out)
                df_progress.to_csv(df_progress_fp, index=False)
        else:
            print(f"{str(af_design_path)} already exists, used cached record...")

    df_out = pd.DataFrame(df_out)
    df_out.to_csv(df_out_fp, index=False)


if __name__ == "__main__":
    typer.run(design)
