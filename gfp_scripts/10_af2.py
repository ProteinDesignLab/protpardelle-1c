from pathlib import Path

from Bio.PDB import MMCIFParser, PDBIO
from colabdesign import clear_mem
from colabdesign.af import mk_af_model
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
import typer


alphabet = "ARNDCQEGHILKMFPSTWYVX"


af2_param_dir = Path("/oak/stanford/groups/possu/tianyu/models/af2/params")


def run_af2(seq_des_csv: Path, num_models: int = 5, num_recycles: int = 3, use_multimer: bool = False, use_templates: bool = False, rm_template_interchain: bool = False, chains: str = "A", homooligomer: bool = False, save_every: int = 1000):
    """
    Given output of caliby, predict structures with AlphaFold2
    """
    df = pd.read_csv(seq_des_csv)
    groups = df["group"].to_list()
    example_ids = df["example_id"].to_list()
    U = df["U"].to_list()
    seqs = df["seq"].to_list()
    pdb_paths = df["out_pdb"].to_list()

    data = []

    for n, (group_name, seq, pdb_path) in tqdm(enumerate(zip(groups, seqs, pdb_paths)), total=len(groups)):
        # output directory name based on group name
        out_dir = seq_des_csv.parent / "af2_pred" / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_fp = out_dir / f"{Path(pdb_path).stem}.pdb"

        # convert .cif to .pdb if needed:
        if Path(pdb_path).suffix == ".cif":
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("struct", str(pdb_path))
            pdb_save_fp = Path(pdb_path).with_suffix(".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_save_fp))
            pdb_path = str(pdb_save_fp)

        af_args = [pdb_path, chains, homooligomer, use_multimer, use_templates]
        if "af_arg_current" not in dir() or af_args != af_arg_current:
            clear_mem()
            af_model = mk_af_model(data_dir=af2_param_dir,
                                    use_multimer=use_multimer,
                                    use_templates=use_templates,
                                    best_metric="plddt")
            af_model.prep_inputs(pdb_path,chains,homooligomer=homooligomer)
            af_arg_current = [x for x in af_args]

        af_model.restart()
        af_model.set_opt("template", rm_ic=rm_template_interchain)
        seq = jnp.array([alphabet.index(aa) for aa in seq])
        af_model.predict(seq=seq,
                        num_recycles=num_recycles,
                        num_models=num_models,
                        verbose=False)
        af_model._save_results(save_best=True, verbose=False)
        af_model.save_current_pdb(output_fp)
        af_model._k += 1

        ret = af_model.aux["log"]
        labels = ["plddt","ptm","rmsd","models"]
        sublist = [ret[label] for label in labels]
        labels.extend(["group","example_id","U","seq","af2_pred_pdb"])
        sublist.extend([groups[n], example_ids[n], U[n], seqs[n], str(output_fp)])

        data.append(sublist)
        if (n+1) % save_every == 0:
            df = pd.DataFrame(data, columns=labels)
            df.to_csv(out_dir / f"{seq_des_csv.stem}_alphafold_results_{n}.csv", index=False)

    df = pd.DataFrame(data, columns=labels)
    df.to_csv(out_dir / f"{seq_des_csv.stem}_alphafold_results_all.csv", index=False)


def runner(seq_des_csv_fp: Path):
    run_af2(seq_des_csv_fp,
            num_models=1,
            num_recycles=3,
            use_multimer=False,
            use_templates=False,
            rm_template_interchain=False,
            chains="A",
            homooligomer=False,
            save_every=500)
    

if __name__ == "__main__":
    typer.run(runner)
