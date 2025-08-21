# Script included for reference only, environment does not have Jax or colabdesign support

import os
import uuid
from pathlib import Path

import pandas as pd
import typer
from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.af import mk_af_model
from colabdesign.shared.utils import copy_dict

alphabet = "ARNDCQEGHILKMFPSTWYVX"
af2_param_dir = "/your/path/to/models/af2/params"
target_dir = Path("/your/path/to/farfalle/motifs/bindcraft")
target_stems = [
    "0_PD-1",
    "1_PD-L1",
    "2_IFNAR2",
    "2LAG",
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
    "A",  # extra 2
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
    "B",  # extra 2
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


def run_af2(
    seq_csv: Path,
    num_models: int = 1,
    num_recycles: int = 1,
    hard_target: bool = False,
    chains: str = "A,B",
    homooligomer: bool = False,
    save_every: int = 1000,
):
    df = pd.read_csv(seq_csv)
    seqs = df["seq"].to_list()
    pdb_paths = df["pdb_path"].to_list()

    out_dir = seq_csv.parent / "af2_pred"
    out_dir.mkdir(exist_ok=True)

    data = []

    for n, (seq, pdb_path) in enumerate(zip(seqs, pdb_paths)):
        complex_output_fp = out_dir / f"{Path(pdb_path).stem}_{n}_complex.pdb"
        binder_output_fp = out_dir / f"{Path(pdb_path).stem}_{n}_binder.pdb"

        af_args = [pdb_path, chains, homooligomer]

        if "af_arg_current" not in dir() or af_args != af_arg_current:
            clear_mem()

            binder_length = len(seq.split(":")[-1])

            target_path = None
            for si, stem in enumerate(target_stems):
                if stem in Path(pdb_path).stem:
                    target_path = str(target_pdb_paths[si].resolve())
                    if hard_target:
                        chain = all_chains[si]
                    else:
                        chain = target_chains[si]
                    binder_chain_id = binder_chains[si]
                    break
            if target_path is None:
                print(f"Target for {str(pdb_path)} not found!")
                raise AssertionError

            # alternative for hard targets:
            if hard_target:
                complex_prediction_model = mk_afdesign_model(
                    protocol="binder",
                    num_recycles=num_recycles,
                    data_dir=af2_param_dir,
                    use_multimer=False,
                    use_initial_guess=True,
                    use_initial_atom_pos=False,
                )
                complex_prediction_model.prep_inputs(
                    pdb_filename=pdb_path,
                    chain=chain,
                    binder_chain=binder_chain_id,
                    binder_len=binder_length,
                    use_binder_template=True,
                    rm_target_seq=False,
                    rm_target_sc=False,
                    rm_template_ic=True,
                )
            else:
                complex_prediction_model = mk_afdesign_model(
                    protocol="binder",
                    num_recycles=num_recycles,
                    data_dir=af2_param_dir,
                    use_multimer=False,
                    use_initial_guess=False,
                    use_initial_atom_pos=False,
                )
                complex_prediction_model.prep_inputs(
                    pdb_filename=target_path,
                    chain=chain,
                    binder_len=binder_length,
                    use_binder_template=False,
                    rm_target_seq=True,
                    rm_target_sc=False,
                )

            unique_id = uuid.uuid4().hex
            os.system(
                f'pdb_selchain -{binder_chain_id} "{pdb_path}" > tmp_{unique_id}.pdb'
            )

            af_model = mk_af_model(
                use_multimer=False,
                use_templates=False,
                best_metric="ptm",
                data_dir=af2_param_dir,
            )
            af_model.prep_inputs(
                f"tmp_{unique_id}.pdb", binder_chain_id, homooligomer=homooligomer
            )
            os.system(f"rm tmp_{unique_id}.pdb")

            af_arg_current = [x for x in af_args]

        labels = ["pdb_path", "seq", "output_path"]
        sublist = [pdb_paths[n], seqs[n], str(complex_output_fp)]
        print(pdb_path)

        # predict binder complex
        # complex_prediction_model.restart()
        binder_sequence = seq.split(":")[-1]
        # binder_sequence = jnp.array([alphabet.index(aa) for aa in binder_sequence])
        for model_num in prediction_models:
            complex_prediction_model.predict(
                seq=binder_sequence,
                models=[model_num],
                num_recycles=num_recycles,
                verbose=False,
            )
            complex_prediction_model.save_pdb(
                complex_output_fp.parent
                / f"{complex_output_fp.stem}_model{model_num}.pdb"
            )

            prediction_metrics = copy_dict(complex_prediction_model.aux["log"])
            prediction_metrics = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in prediction_metrics.items()
            }

            metric_names = ["plddt", "ptm", "i_ptm", "pae", "i_pae"]
            sublist.extend([prediction_metrics[label] for label in metric_names])
            labels.extend([f"model_{model_num}_complex_{m}" for m in metric_names])

            print(f"Model {model_num} Complex Metrics")
            for pk in metric_names:
                print(pk, prediction_metrics[pk])

        # predict binder alone
        # af_model.restart()
        for model_num in prediction_models:
            af_model.predict(
                seq=binder_sequence,
                models=[model_num],
                num_recycles=num_recycles,
                verbose=False,
            )
            af_model.save_pdb(
                binder_output_fp.parent
                / f"{binder_output_fp.stem}_model{model_num}.pdb"
            )

            prediction_metrics = copy_dict(af_model.aux["log"])
            prediction_metrics = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in prediction_metrics.items()
            }

            metric_names = ["plddt", "ptm", "pae", "rmsd"]
            sublist.extend([prediction_metrics[label] for label in metric_names])
            labels.extend([f"model_{model_num}_binder_{m}" for m in metric_names])

            print(f"Model {model_num} Binder Metrics:")
            for pk in metric_names:
                print(pk, prediction_metrics[pk])

        data.append(sublist)
        if (n + 1) % save_every == 0:
            df = pd.DataFrame(data, columns=labels)
            df.to_csv(out_dir / f"alphafold_results_{n}.csv", index=False)

    df = pd.DataFrame(data, columns=labels)
    df.to_csv(out_dir / "alphafold_results_all.csv", index=False)


if __name__ == "__main__":
    typer.run(run_af2)
