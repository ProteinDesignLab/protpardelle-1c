"""Entrypoint for Protpardelle-1c sampling.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import itertools
import logging
import math
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import shutil
import uuid

import hydra
import numpy as np
import pandas as pd
import torch
import typer
import wandb
from Bio import SeqIO
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchtyping import TensorType
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.core.models import Protpardelle, load_model
from protpardelle.data.atom import atom37_coords_from_bb
from protpardelle.data.dataset import make_fixed_size_1d
from protpardelle.data.motif import contig_to_motif_placement
from protpardelle.data.pdb_io import write_coords_to_pdb
from protpardelle.data.sequence import seq_to_aatype
from protpardelle.env import (
    FOLDSEEK_BIN,
    PACKAGE_ROOT_DIR,
    PROTEINMPNN_WEIGHTS,
    PROTPARDELLE_MODEL_PARAMS,
    PROTPARDELLE_OUTPUT_DIR,
)
from protpardelle.evaluate import compute_self_consistency
from protpardelle.integrations import protein_mpnn
from protpardelle.utils import (
    apply_dotdict_recursively,
    get_default_device,
    seed_everything,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def save_samples(
    aux,
    save_dir,
    save_name,
    bb_only: bool = True,
    allatom: bool = False,
    motif_placements: list[str] | None = None,
):
    (
        sampled_coords,
        trimmed_residue_index,
        trimmed_chain_index,
        seq_mask,
        samp_aux,
        sc_aux,
    ) = aux
    if sc_aux is not None:
        pred_coords = sc_aux["pred"]
        designed_seqs = sc_aux["seqs"]
        all_atom_plddts = sc_aux["all_atom_plddt"]
        plddts = sc_aux["plddt"]
        paes = sc_aux["pae"]
        scrmsds = (
            sc_aux["allatom_scaffold_scrmsd"]
            if allatom
            else sc_aux["ca_scaffold_scrmsd"]
        )
        motif_ca_rmsds = (
            sc_aux["ca_motif_sample_rmsd"] if allatom else sc_aux["ca_motif_pred_rmsd"]
        )
        motif_allatom_rmsds = (
            sc_aux["allatom_motif_sample_rmsd"]
            if allatom
            else sc_aux["allatom_motif_pred_rmsd"]
        )

    all_samp_save_names = []

    for idx, _ in enumerate(sampled_coords):

        if allatom:
            dummy_aatype = samp_aux["s"]
        else:
            dummy_aatype = (
                seq_mask[idx][seq_mask[idx] == 1] * residue_constants.restype_order["G"]
            ).long()
            dummy_aatype = torch.tile(
                dummy_aatype.unsqueeze(0), dims=(len(sampled_coords), 1)
            )

        if samp_aux["motif_idx"] is not None and len(samp_aux["motif_idx"]) > 0:
            for ii, mi in enumerate(samp_aux["motif_idx"][idx]):
                dummy_aatype[idx][mi] = samp_aux["motif_aatype"][idx][ii]

        if sampled_coords[idx].shape[-2] == 4:
            coords_to_save = atom37_coords_from_bb(sampled_coords[idx])
        else:
            coords_to_save = sampled_coords[idx]
        samp_save_name = save_dir / f"sample_{save_name}_{idx}.pdb"

        if bb_only:
            coords_to_save = coords_to_save[:, (0, 1, 2, 4), :]

        write_coords_to_pdb(
            coords_to_save,
            samp_save_name,
            batched=False,
            aatype=dummy_aatype[idx],
            residue_index=trimmed_residue_index[idx],
            chain_index=trimmed_chain_index[idx],
        )
        all_samp_save_names.append(str(samp_save_name.resolve()))

    all_pred_save_names = []

    if sc_aux is not None:

        num_designs_per_structure = len(designed_seqs) // len(sampled_coords)

        Path(f"{save_dir}/esmfold").mkdir(parents=True, exist_ok=True)

        for idx, _ in enumerate(designed_seqs):
            designed_seq = seq_to_aatype(designed_seqs[idx].replace(":", ""))
            scaffold_idx = idx // num_designs_per_structure
            if motif_placements is not None:
                full_save_name = f"{save_name}_scrmsd{scrmsds[idx]:.2f}_motifcarmsd{motif_ca_rmsds[idx]:.2f}_motifaarmsd{motif_allatom_rmsds[idx]:.2f}_pae{paes[idx]:.2f}_plddt{plddts[idx]:.2f}_bb_pred_{scaffold_idx}".replace(
                    ".", "-"
                )
            else:
                full_save_name = f"{save_name}_scrmsd{scrmsds[idx]:.2f}_pae{paes[idx]:.2f}_plddt{plddts[idx]:.2f}_bb_pred_{scaffold_idx}".replace(
                    ".", "-"
                )
            pred_save_name = f"{save_dir}/esmfold/{full_save_name}"
            pred_save_name = f"{pred_save_name}.pdb"
            all_pred_save_names.append(pred_save_name)
            write_coords_to_pdb(
                pred_coords[idx],
                pred_save_name,
                batched=False,
                aatype=designed_seq,
                residue_index=trimmed_residue_index[scaffold_idx],
                chain_index=trimmed_chain_index[scaffold_idx],
                b_factors=all_atom_plddts[idx],
            )

    return all_samp_save_names, all_pred_save_names


def draw_samples(
    model: Protpardelle,
    seq_mask: TensorType["b n", float] | None = None,
    residue_index: TensorType["b n", float] | None = None,
    chain_index: TensorType["b n", int] | None = None,
    hotspots: str | list[str] | None = None,
    sse_cond: TensorType["b n", int] | None = None,
    adj_cond: TensorType["b n n", int] | None = None,
    motif_placements_full: list[str] | None = None,
    num_samples: int | None = None,
    length_ranges_per_chain: list[tuple[int, int]] = [(50, 512)],
    return_aux: bool = False,
    return_sampling_runtime: bool = False,
    return_coords_and_aux: bool = False,
    **sampling_kwargs,
):

    device = model.device
    if seq_mask is None:
        seq_mask, residue_index, chain_index = model.make_seq_mask_for_sampling(
            length_ranges_per_chain=length_ranges_per_chain, num_samples=num_samples
        )

    start = time.time()

    allatom_cfg = apply_dotdict_recursively(sampling_kwargs["allatom_cfg"])
    stage2_cfg = apply_dotdict_recursively(sampling_kwargs["stage2_cfg"])
    sampling_kwargs.pop("allatom_cfg")
    sampling_kwargs.pop("stage2_cfg")
    sampling_kwargs.update(allatom_cfg)

    aux = model.sample(
        seq_mask=seq_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        hotspots=hotspots,
        sse_cond=sse_cond,
        adj_cond=adj_cond,
        motif_placements_full=motif_placements_full,
        return_last=False,
        return_aux=True,
        dummy_fill_mode=model.config.data.dummy_fill_mode,
        **sampling_kwargs,
    )
    # account for possible override from partial diffusion input structure
    residue_index = aux["residue_index"]
    chain_index = aux["chain_index"]

    # Stage 2 sampling is allatom partial diffusion given sequence from stage1 for more explicit sidechain-driven backbone change
    stage2_enabled = stage2_cfg.pop("enabled")
    if stage2_enabled:
        rewind_steps = stage2_cfg.pop("rewind_steps")
        sampling_kwargs.update(stage2_cfg)

        samp_seq = aux["st_traj"][-1]
        samp_coords = aux["xt_traj"][-1]

        # temp directory for stage1 outputs / inputs for stage 2
        unique_id = uuid.uuid4().hex
        tmp_dir = Path(f"tmp-{unique_id}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for i, _ in enumerate(samp_coords):
            length_bi = int(seq_mask[i].sum().item())
            write_coords_to_pdb(
                samp_coords[i][:length_bi],
                str(tmp_dir / f"stage1_{i}.pdb"),
                batched=False,
                aatype=samp_seq[i][:length_bi],
                chain_index=chain_index[i][:length_bi],
            )

        start = time.time()
        sampling_kwargs["jump_steps"] = False
        sampling_kwargs["uniform_steps"] = True

        # Here, an alternative strategy is to set stage2 as partial diffusion conditioned on backbone
        # uncomment the subsequent lines to achieve this
        # sampling_kwargs['conditional_cfg']['enabled'] = False
        # sampling_kwargs['conditional_cfg']['crop_conditional_guidance']['enabled'] = False
        # sampling_kwargs['conditional_cfg']['crop_conditional_guidance']['strategy'] = 'backbone'
        # sampling_kwargs['conditional_cfg']['reconstruction_guidance']['enabled'] = False
        # sampling_kwargs['conditional_cfg']['replacement_guidance']['enabled'] = False

        sampling_kwargs["partial_diffusion"]["enabled"] = True
        sampling_kwargs["partial_diffusion"]["n_steps"] = rewind_steps
        sampling_kwargs["partial_diffusion"]["pdb_file_path"] = [
            tmp_dir / f"stage1_{i}.pdb" for i in range(len(samp_coords))
        ]
        stage2_aux = model.sample(
            gt_aatype=samp_seq.to(device),
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            hotspots=hotspots,
            sse_cond=sse_cond,
            adj_cond=adj_cond,
            motif_placements_full=motif_placements_full,
            return_last=False,
            return_aux=True,
            dummy_fill_mode=model.config.data.dummy_fill_mode,
            motif_all_atom_stage1=aux[
                "motif_all_atom"
            ],  # use the centering + random rotation sampled in stage1
            motif_idx_stage1=aux["motif_idx"],
            stage2=True,
            **sampling_kwargs,
        )
        stage2_aux.pop("s")

        stage2_keys = list(stage2_aux.keys())
        for k in stage2_keys:
            stage2_aux[f"stage2_{k}"] = stage2_aux.pop(k)
        aux = {**aux, **stage2_aux}

        shutil.rmtree(tmp_dir)

    aux["runtime"] = time.time() - start
    seq_lens = seq_mask.sum(-1).long()

    xt_traj_key = "stage2_xt_traj" if stage2_enabled else "xt_traj"
    if model.config.model.task != "backbone" or (
        "save_motif_sidechain" in sampling_kwargs["conditional_cfg"]
        and sampling_kwargs["conditional_cfg"]["save_motif_sidechain"]
    ):
        cropped_samp_coords = [
            s[: seq_lens[i], :] for i, s in enumerate(aux[xt_traj_key][-1])
        ]
    else:
        cropped_samp_coords = [
            s[: seq_lens[i], model.bb_idxs] for i, s in enumerate(aux[xt_traj_key][-1])
        ]
    cropped_residue_index = [r[: seq_lens[i]] for i, r in enumerate(residue_index)]
    cropped_chain_index = [c[: seq_lens[i]] for i, c in enumerate(chain_index)]

    if return_aux:
        return aux
    else:
        if return_sampling_runtime:
            return cropped_samp_coords, cropped_residue_index, cropped_chain_index, seq_mask, aux["runtime"]
        elif return_coords_and_aux:
            return cropped_samp_coords, cropped_residue_index, cropped_chain_index, seq_mask, aux
        else:
            return cropped_samp_coords, cropped_residue_index, cropped_chain_index, seq_mask


def generate(
    model,
    sampling_config,
    num_samples=8,
    num_mpnn_seqs=8,
    length_ranges_per_chain=None,
    all_lengths=None,
    batch_size=32,
    fixed_motif_pos=None,
    motif_placements_full=None,
    hotspots=None,
    sse_cond=None,
    adj_cond=None,
    run_name="",
    allatom=False,
):
    device = get_default_device()
    if num_mpnn_seqs > 0:
        mpnn_model = protein_mpnn.get_mpnn_model(
            PROTEINMPNN_WEIGHTS,
            device=device,
        )
    else:
        mpnn_model = None

    trimmed_coords = []
    trimmed_residue_index = []
    trimmed_chain_index = []
    seq_mask = []
    atom_mask = []
    motif_idx = []
    motif_coords = []
    motif_aatypes = []
    motif_atom_mask = []
    sequences = []
    runtime = 0

    seq_mask_input, residue_index_input, chain_index_input = None, None, None
    all_seq_mask, all_residue_index, all_chain_index = [], [], []
    if all_lengths is not None:
        max_length = np.array(all_lengths).sum(1).max()
        for curr_len in all_lengths:
            curr_length_ranges_per_chain = []
            for curr_chain_len in curr_len:
                curr_length_ranges_per_chain.append([curr_chain_len, curr_chain_len])
            seq_mask_in, residue_index_in, chain_index_in = (
                model.make_seq_mask_for_sampling(
                    length_ranges_per_chain=curr_length_ranges_per_chain, num_samples=1
                )
            )
            all_seq_mask.append(
                make_fixed_size_1d(seq_mask_in.flatten(), fixed_size=max_length)[0]
            )
            all_residue_index.append(
                make_fixed_size_1d(residue_index_in.flatten(), fixed_size=max_length)[0]
            )
            all_chain_index.append(
                make_fixed_size_1d(chain_index_in.flatten(), fixed_size=max_length)[0]
            )
        seq_mask_input = torch.stack(all_seq_mask)
        residue_index_input = torch.stack(all_residue_index)
        chain_index_input = torch.stack(all_chain_index)

    batch_sizes = [batch_size] * (num_samples // batch_size)
    if num_samples % batch_size != 0:
        batch_sizes.append(num_samples % batch_size)

    for i, bs in enumerate(batch_sizes):

        si, ei = i * bs, (i + 1) * bs
        if i == len(batch_sizes) - 1:
            si, ei = -bs, num_samples

        if fixed_motif_pos is not None:
            sampling_config["sampling"]["conditional_cfg"][
                "discontiguous_motif_assignment"
            ]["fixed_motif_pos"] = fixed_motif_pos[si:ei]

        if sse_cond is not None and adj_cond is not None:
            sse_cond = torch.cat([sse_cond.clone() for _ in range(bs)], dim=0)
            adj_cond = torch.cat([adj_cond.clone() for _ in range(bs)], dim=0)

        curr_sampling_config = deepcopy(sampling_config["sampling"])

        trimmed_coords_bi, trimmed_residue_index_bi, trimmed_chain_index_bi, seq_mask_bi, samp_aux_bi = (
            draw_samples(
                model,
                num_samples=bs,
                seq_mask=seq_mask_input[si:ei] if seq_mask_input is not None else None,
                residue_index=(
                    residue_index_input[si:ei]
                    if residue_index_input is not None
                    else None
                ),
                chain_index=(
                    chain_index_input[si:ei] if chain_index_input is not None else None
                ),
                length_ranges_per_chain=length_ranges_per_chain,
                return_coords_and_aux=True,
                hotspots=hotspots,
                sse_cond=sse_cond,
                adj_cond=adj_cond,
                motif_placements_full=(
                    motif_placements_full[si:ei]
                    if motif_placements_full is not None
                    else None
                ),
                **curr_sampling_config,
            )
        )
        trimmed_coords.extend(trimmed_coords_bi)
        trimmed_residue_index.extend(trimmed_residue_index_bi)
        trimmed_chain_index.extend(trimmed_chain_index_bi)
        seq_mask.extend(seq_mask_bi)
        if samp_aux_bi["motif_idx"] is not None:
            motif_idx.extend(samp_aux_bi["motif_idx"])
            motif_aatypes.extend(samp_aux_bi["motif_aatype"])
            motif_coords.append(samp_aux_bi["motif_all_atom"])
            motif_atom_mask.extend(samp_aux_bi["motif_atom_mask"])
        runtime += samp_aux_bi["runtime"]
        sequences.extend(samp_aux_bi["s"])
        atom_mask.extend(samp_aux_bi["atom_mask"])

    motif_coords = torch.cat(motif_coords, dim=0) if motif_coords else None
    if not motif_aatypes:
        motif_idx = None
        motif_aatypes = None
        motif_atom_mask = None

    samp_aux = {
        "atom_mask": atom_mask,
        "motif_coords": motif_coords,
        "motif_idx": motif_idx,
        "motif_aatype": motif_aatypes,
        "motif_atom_mask": motif_atom_mask,
        "s": sequences,
        "runtime": runtime,
    }

    sampled_sequences = None
    if allatom:
        sampled_sequences = [
            "".join([residue_constants.restypes[aaint] for aaint in s_hat_bi])
            for s_hat_bi in sequences
        ]
    # just need motif_idx and motif_aatype from samp_aux, don't need to keep anything else
    sc_aux = None
    if num_mpnn_seqs > 0:
        sc_aux = compute_self_consistency(
            trimmed_coords,
            trimmed_chain_index=trimmed_chain_index,
            mpnn_model=mpnn_model,
            num_seqs=num_mpnn_seqs,
            motif_idx=motif_idx,
            motif_coords=motif_coords,
            motif_aatypes=motif_aatypes,
            sampled_sequences=sampled_sequences,
            tmp_prefix=run_name,
            allatom=allatom,
            atom_mask=atom_mask,
            motif_atom_mask=motif_atom_mask,
        )

    return (trimmed_coords, trimmed_residue_index, trimmed_chain_index, seq_mask, samp_aux, sc_aux)


def sample(
    sampling_yaml_path: Path,
    project_name: str = "protpardelle-1c-sampling",
    motif_dir: Path = Path("motifs/nanobody"),
    num_samples: int = 8,
    num_mpnn_seqs: int = 8,
    batch_size: int = 32,
    save_shortname: bool = True,
    seed: int | None = None,
    use_wandb: bool = False,
    array_id: int | None = None,
    num_arrays: int | None = None,
) -> list[Path]:
    """Sampling with Protpardelle-1c.

    Args:
        sampling_yaml_path (Path): Path to sampling config, see examples/sampling/*.yaml for examples
        project_name (str, optional): Name of project for wandb. Defaults to "protpardelle-1c-sampling".
        motif_dir (Path, optional): Folder containing motifs to scaffold. Defaults to Path("motifs/nanobody").
        num_samples (int, optional): Total number of samples to draw. Defaults to 8.
        num_mpnn_seqs (int, optional): If 0, skips sequence design and ESMFold evaluation. Defaults to 8.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        seed (int | None, optional): Random seed. Defaults to None.
        use_wandb (bool, optional): If True, use wandb to log results. Defaults to False.
        array_id (int | None, optional): Slurm array id for parallelization. Defaults to None.
        num_arrays (int | None, optional): Number of arrays for parallelization. Defaults to None.
    """

    if seed is not None:
        seed_everything(seed)

    run_name = sampling_yaml_path.stem

    save_dir = PROTPARDELLE_OUTPUT_DIR / f"{run_name}"
    save_dir.mkdir(exist_ok=True, parents=True)

    with initialize_config_dir(
        config_dir=str(sampling_yaml_path.parent.resolve()), version_base="1.3.2"
    ):
        runner_cfg = compose(config_name=run_name)

    runner_cfg = hydra.utils.call(runner_cfg)
    runner_cfg = OmegaConf.to_container(runner_cfg, resolve=True)

    search_space = runner_cfg["search_space"]

    partial_diffusion = (
        "partial_diffusion" in runner_cfg and runner_cfg["partial_diffusion"]["enabled"]
    )
    if partial_diffusion:
        search_space["rewind_steps"] = runner_cfg["partial_diffusion"]["rewind_steps"]

    motif_fps = []
    motif_contigs = []
    scaffold_lengths = []
    hotspots = []
    ssadj_fps = []
    all_save_dirs = []

    for ri, (motif_cfg, motif_contig, length_range, hs, ssadj) in enumerate(
        zip(
            runner_cfg["motifs"],
            runner_cfg["motif_contigs"],
            runner_cfg["total_lengths"],
            runner_cfg["hotspots"],
            runner_cfg["ssadj"],
        )
    ):
        if motif_cfg is None:
            motif_cfg = f"{ri:03}_unconditional"
        if not motif_cfg.endswith('.pdb') and not motif_cfg.endswith('.cif'):
            motif_fps.append(motif_dir / f"{motif_cfg}.pdb")
        else:
            motif_fps.append(motif_dir / motif_cfg)
        motif_contigs.append(motif_contig)
        scaffold_lengths.append(length_range)
        hotspots.append(hs)
        ssadj_fps.append(ssadj)

    # get all params, and optionally split into chunks for parallelization
    all_params = list(itertools.product(*search_space.values()))
    if array_id is not None:
        chunk_size = math.ceil(len(all_params) / num_arrays)
        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_params))
        all_params = all_params[start_idx:end_idx]

    # for a given setting
    for curr_params in tqdm(all_params, "Evaluating search space"):

        if partial_diffusion:
            (
                (model_name, epoch, sampling_config_name),
                stepscale,
                schurn,
                cc_start,
                (dx, dy, dz),
                rs,
            ) = curr_params
        else:
            (
                (model_name, epoch, sampling_config_name),
                stepscale,
                schurn,
                cc_start,
                (dx, dy, dz),
            ) = curr_params
            rs = None

        dxs = f"{dx:.1f}"
        dys = f"{dy:.1f}"
        dzs = f"{dz:.1f}"

        save_suffix = f"{model_name}-epoch{epoch}-{sampling_config_name}-ss{stepscale}-schurn{schurn}-ccstart{cc_start}-dx{dxs}-dy{dys}-dz{dzs}-rewind{rs}"

        per_config_save_dir = save_dir / save_suffix  # one config, all motifs
        per_config_save_dir.mkdir(exist_ok=True)

        with initialize_config_dir(
            config_dir=str(PACKAGE_ROOT_DIR / "configs/sampling"),
            version_base="1.3.2",
        ):
            sampling_config = compose(config_name=sampling_config_name)

        sampling_config = hydra.utils.call(sampling_config)
        sampling_config = OmegaConf.to_container(sampling_config, resolve=True)

        if use_wandb and num_mpnn_seqs > 0:
            wandb.init(
                project=project_name,
                name=save_suffix,
                config=sampling_config,
            )

        all_samp_save_names = []
        all_fix_pos = []
        df_metrics = []

        start_time = time.time()
        total_sampling_time = 0

        model_info = None, None

        for (
            motif_fp,
            motif_contig,
            length_ranges_per_chain,
            hotspot,
            ssadj_fp,
        ) in zip(
            motif_fps,
            motif_contigs,
            scaffold_lengths,
            hotspots,
            ssadj_fps,
        ):
            if save_shortname:
                save_name = f"{motif_fp.stem}"

            per_motif_save_dir = (
                per_config_save_dir / motif_fp.stem
            )  # one config, one motif
            per_motif_save_dir.mkdir(exist_ok=True)
            all_save_dirs.append(per_motif_save_dir)

            config_path = str(
                PROTPARDELLE_MODEL_PARAMS / "configs" / f"{model_name}.yaml"
            )
            checkpoint_path = str(
                PROTPARDELLE_MODEL_PARAMS / "weights" / f"{model_name}_epoch{epoch}.pth"
            )

            sampling_config["sampling"]["step_scale"] = stepscale
            sampling_config["sampling"]["s_churn"] = schurn
            sampling_config["sampling"]["conditional_cfg"]["crop_conditional_guidance"][
                "start"
            ] = cc_start
            if partial_diffusion:
                shutil.copy(motif_fp, per_motif_save_dir)
                sampling_config["sampling"]["partial_diffusion"][
                    "pdb_file_path"
                ] = motif_fp
                sampling_config["sampling"]["partial_diffusion"]["n_steps"] = rs
                sampling_config["sampling"]["motif_file_path"] = "test_dir/empty.pdb"
            else:
                sampling_config["sampling"]["motif_file_path"] = motif_fp
            sampling_config["sampling"]["dx"] = dx
            sampling_config["sampling"]["dy"] = dy
            sampling_config["sampling"]["dz"] = dz

            # parse motif contig
            motif_idx, motif_placements, motif_placements_full, all_lengths = (
                None,
                None,
                None,
                None,
            )
            if (
                motif_contig is not None
                and motif_contig != "partial_diffusion"
                and motif_contig != "pdb"
                and motif_contig != "dynamic"
                and not Path(motif_contig).exists()
                and not motif_contig.startswith("seq/")
            ):
                sampling_config["sampling"]["conditional_cfg"][
                    "discontiguous_motif_assignment"
                ]["strategy"] = "fixed"
                # Do for each chain
                chain_motif_contigs = motif_contig.split(";/;")
                assert len(chain_motif_contigs) == len(
                    length_ranges_per_chain
                ), f"Contig {motif_contig} has {len(chain_motif_contigs)} chains but length ranges specify {len(length_ranges_per_chain)} chains."
                prev_length = 0
                for ci, chain_motif_contig in enumerate(chain_motif_contigs):
                    (
                        chain_motif_idx,
                        chain_motif_placements,
                        chain_motif_placements_full,
                        chain_lengths,
                    ) = contig_to_motif_placement(
                        chain_motif_contig, length_ranges_per_chain[ci], num_samples
                    )
                    if motif_idx is None:
                        motif_idx = chain_motif_idx
                        motif_placements = chain_motif_placements
                        motif_placements_full = chain_motif_placements_full
                        all_lengths = [[cl] for cl in chain_lengths]
                    else:
                        for bi, _ in enumerate(chain_motif_idx):
                            motif_idx[bi].extend(
                                list(np.array(chain_motif_idx[bi]) + prev_length)
                            )
                            motif_placements[bi] = (
                                motif_placements[bi]
                                + ";/;"
                                + chain_motif_placements[bi]
                            )
                            motif_placements_full[bi] = (
                                motif_placements_full[bi]
                                + ";/;"
                                + chain_motif_placements_full[bi]
                            )
                            all_lengths[bi].append(chain_lengths[bi])
                    prev_length += length_ranges_per_chain[ci][0]

            if motif_contig is not None and motif_contig.startswith("seq/"):
                repack_seq = motif_contig.split("/")[-1]
                sampling_config["sampling"]["partial_diffusion"]["seq"] = repack_seq

            if model_info == (None, None) or model_info != (
                config_path,
                checkpoint_path,
            ):
                model = load_model(config_path, checkpoint_path)
                model_info = (config_path, checkpoint_path)

            allatom = (
                sampling_config["sampling"]["allatom_cfg"]["jump_steps"]
                or sampling_config["sampling"]["allatom_cfg"]["uniform_steps"]
            )
            if allatom:
                mpnn_ckpt_path = str(
                    PROTPARDELLE_MODEL_PARAMS / "weights" / "cc58_epoch97_minimpnn.pth"
                )
                model.load_minimpnn(mpnn_ckpt_path=mpnn_ckpt_path)

            sse_cond = None
            adj_cond = None
            if ssadj_fp is not None:
                sse_cond = (
                    torch.from_numpy(
                        torch.load(motif_dir / f"{ssadj_fp[0]}.pt", weights_only=False)
                    )
                    .long()
                    .unsqueeze(0)
                )
                adj_cond = (
                    torch.load(motif_dir / f"{ssadj_fp[1]}.pt", weights_only=False)
                    .long()
                    .unsqueeze(0)
                )

            aux = generate(
                model,
                sampling_config,
                num_samples=num_samples,
                num_mpnn_seqs=num_mpnn_seqs,
                length_ranges_per_chain=length_ranges_per_chain,
                all_lengths=all_lengths,
                batch_size=batch_size,
                fixed_motif_pos=motif_idx,
                motif_placements_full=motif_placements_full,
                hotspots=hotspot,
                sse_cond=sse_cond,
                adj_cond=adj_cond,
                run_name=run_name,
                allatom=allatom,
            )

            curr_runtime = aux[-2]["runtime"]
            total_sampling_time += curr_runtime

            logger.info("Sampling %s took %.2f seconds.", motif_fp.stem, curr_runtime)

            # save motif placements
            df_scaffold_info = defaultdict(list)

            df_scaffold_info["sample_num"] = list(range(num_samples))
            if motif_contig is not None:
                df_scaffold_info["motif_placements"] = motif_placements

            df_scaffold_info = pd.DataFrame(df_scaffold_info)
            df_scaffold_info.to_csv(
                per_motif_save_dir / "scaffold_info.csv", index=False
            )

            curr_samp_save_names, curr_pred_save_names = save_samples(
                aux,
                per_motif_save_dir,
                save_name,
                bb_only=model.config.model.task == "backbone",
                allatom=allatom,
                motif_placements=motif_placements_full,
            )

            curr_fix_pos = []
            if motif_idx is not None:
                curr_fix_pos.extend(
                    " ".join([str(mi + 1) for mi in motif_idx_bi])
                    for motif_idx_bi in motif_idx
                )
                all_fix_pos.extend(curr_fix_pos)
            else:
                all_fix_pos.extend([""] * len(curr_samp_save_names))

            all_samp_save_names.extend(curr_samp_save_names)

            if num_mpnn_seqs > 0:
                sc_aux = aux[-1]
                sc_aux.pop("all_atom_plddt")
                sc_aux.pop("pred")
                for k, v in sc_aux.items():
                    converted = []
                    for entry in v:
                        if isinstance(entry, torch.Tensor):
                            converted.append(entry.cpu().numpy())
                        else:
                            converted.append(entry)
                    sc_aux[k] = converted

                metrics = pd.DataFrame(sc_aux)
                metrics["motif_name"] = [motif_fp.stem] * len(metrics)
                metrics["save_name"] = curr_pred_save_names
                for pi, param_name in enumerate(search_space.keys()):
                    metrics[param_name] = [curr_params[pi]] * len(metrics)

                best_per_structure_metric_dfs = []

                for _, per_structure_metrics in metrics.groupby("structure_index"):
                    if motif_contig is None:
                        if allatom:
                            per_structure_df_best = per_structure_metrics[
                                (per_structure_metrics["allatom_scaffold_scrmsd"] < 2.0)
                            ]
                            if not per_structure_df_best.empty:
                                per_structure_df_best = per_structure_df_best.loc[
                                    [
                                        per_structure_df_best[
                                            "allatom_scaffold_scrmsd"
                                        ].idxmin()
                                    ]
                                ]
                        else:
                            per_structure_df_best = per_structure_metrics[
                                (per_structure_metrics["ca_scaffold_scrmsd"] < 2.0)
                            ]
                            if not per_structure_df_best.empty:
                                per_structure_df_best = per_structure_df_best.loc[
                                    [
                                        per_structure_df_best[
                                            "ca_scaffold_scrmsd"
                                        ].idxmin()
                                    ]
                                ]

                    elif allatom:
                        per_structure_df_best = per_structure_metrics[
                            (per_structure_metrics["ca_motif_sample_rmsd"] < 1.0)
                            & (
                                per_structure_metrics["allatom_motif_sample_rmsd"]
                                < 2.0
                            )
                            & (
                                per_structure_metrics["allatom_scaffold_scrmsd"]
                                < 2.0
                            )
                        ]
                        if not per_structure_df_best.empty:
                            per_structure_df_best = per_structure_df_best.loc[
                                [
                                    per_structure_df_best[
                                        "allatom_motif_sample_rmsd"
                                    ].idxmin()
                                ]
                            ]
                    else:
                        per_structure_df_best = per_structure_metrics[
                            (per_structure_metrics["ca_motif_sample_rmsd"] < 1.0)
                            & (
                                per_structure_metrics["allatom_motif_pred_rmsd"]
                                < 1.0
                            )  # More strict
                            & (per_structure_metrics["ca_scaffold_scrmsd"] < 2.0)
                        ]
                        if not per_structure_df_best.empty:
                            per_structure_df_best = per_structure_df_best.loc[
                                [
                                    per_structure_df_best[
                                        "allatom_motif_pred_rmsd"
                                    ].idxmin()
                                ]
                            ]
                    if not per_structure_df_best.empty:
                        best_per_structure_metric_dfs.append(per_structure_df_best)

                # compute number of unique successes
                num_success = len(best_per_structure_metric_dfs)
                if num_success > 0:
                    best_per_structure_metric_df = pd.concat(
                        best_per_structure_metric_dfs
                    )

                    self_consistent_fps = best_per_structure_metric_df[
                        "save_name"
                    ].to_list()
                    # copy self-consistent structures to its own folder
                    self_consistent_dir = (
                        per_motif_save_dir / "esm_successful_structures"
                    )
                    if self_consistent_dir.exists():
                        shutil.rmtree(self_consistent_dir)
                    self_consistent_dir.mkdir(exist_ok=True, parents=True)

                    for sc_fp in self_consistent_fps:
                        shutil.copy(sc_fp, self_consistent_dir)

                    # Foldseek clustering command from La-Proteina
                    foldseek_dir = per_motif_save_dir / "foldseek"
                    foldseek_cmd = f"{FOLDSEEK_BIN} easy-cluster {self_consistent_dir} {foldseek_dir}/res {foldseek_dir} --alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5 --single-step-clustering > /dev/null 2>&1"
                    os.system(foldseek_cmd)

                    # parse Foldseek output for unique successes
                    cluster_fp = foldseek_dir / "res_rep_seq.fasta"
                    num_unique_successes = 0
                    for _ in SeqIO.parse(cluster_fp, "fasta"):
                        num_unique_successes += 1

                else:
                    num_unique_successes = num_success

                redundant_success_rate = num_success / num_samples
                nonredundant_success_rate = num_unique_successes / num_samples

                logger.info("Redundant success rate: %d/%d", num_success, num_samples)
                logger.info(
                    "Non-redundant success rate: %d/%d",
                    num_unique_successes,
                    num_samples,
                )

                metrics["redundant_success_rate"] = [redundant_success_rate] * len(
                    metrics
                )
                metrics["nonredundant_success_rate"] = [
                    nonredundant_success_rate
                ] * len(metrics)

                df_metrics.append(metrics)
                metrics.to_csv(per_motif_save_dir / "esm_metrics.csv", index=False)

                df_scaffold_info = pd.DataFrame(df_scaffold_info)
                df_scaffold_info.to_csv(
                    per_motif_save_dir / "scaffold_info.csv", index=False
                )

                log_dict = {
                    "redundant_success_rate": redundant_success_rate,
                    "nonredundant_success_rate": nonredundant_success_rate,
                    "redundant_success_count": num_success,
                    "nonredundant_success_count": num_unique_successes,
                }
                if allatom:
                    log_dict["ca_motif_sample_rmsd_best"] = metrics[
                        "ca_motif_sample_rmsd"
                    ].min()
                    log_dict["ca_motif_sample_rmsd_mean"] = metrics[
                        "ca_motif_sample_rmsd"
                    ].mean()
                    log_dict["allatom_motif_sample_rmsd_best"] = metrics[
                        "allatom_motif_sample_rmsd"
                    ].min()
                    log_dict["allatom_motif_sample_rmsd_mean"] = metrics[
                        "allatom_motif_sample_rmsd"
                    ].mean()
                    log_dict["allatom_scaffold_scrmsd_best"] = metrics[
                        "allatom_scaffold_scrmsd"
                    ].min()
                    log_dict["allatom_scaffold_scrmsd_mean"] = metrics[
                        "allatom_scaffold_scrmsd"
                    ].mean()
                else:
                    log_dict["ca_motif_pred_rmsd_best"] = metrics[
                        "ca_motif_pred_rmsd"
                    ].min()
                    log_dict["ca_motif_pred_rmsd_mean"] = metrics[
                        "ca_motif_pred_rmsd"
                    ].mean()
                    log_dict["allatom_motif_pred_rmsd_best"] = metrics[
                        "allatom_motif_pred_rmsd"
                    ].min()
                    log_dict["allatom_motif_pred_rmsd_mean"] = metrics[
                        "allatom_motif_pred_rmsd"
                    ].mean()
                    log_dict["ca_scaffold_scrmsd_best"] = metrics[
                        "ca_scaffold_scrmsd"
                    ].min()
                    log_dict["ca_scaffold_scrmsd_mean"] = metrics[
                        "ca_scaffold_scrmsd"
                    ].mean()

                if use_wandb and num_mpnn_seqs > 0:
                    wandb.log(log_dict)

        time_elapsed = time.time() - start_time

        logger.info("Sampling concluded after %.2f seconds.", time_elapsed)
        logger.info(
            "Of this, %.2f seconds were for actual sampling.", total_sampling_time
        )
        logger.info("%d total samples were drawn.", num_samples * len(motif_fps))

        df_samp_info = pd.DataFrame()
        df_samp_info["pdb_path"] = all_samp_save_names
        df_samp_info["fix_pos"] = all_fix_pos
        df_samp_info.to_csv(per_config_save_dir / "design_input.csv", index=False)
    
    return all_save_dirs


@app.command()
def main(
    project_name: str = typer.Option(
        "protpardelle-1c-sampling", help="wandb project name"
    ),
    motif_dir: Path = typer.Option(
        Path("motifs/nanobody"), help="Directory containing motif PDBs"
    ),
    num_samples: int = typer.Option(8, help="Number of samples to draw"),
    num_mpnn_seqs: int = typer.Option(
        8, help="Number of sequences to design with MPNN (0 to skip)"
    ),
    batch_size: int = typer.Option(32, help="Batch size for sampling"),
    save_shortname: bool = typer.Option(
        True, help="Whether to save with short names (motif name only)"
    ),
    seed: int | None = typer.Option(None, help="Random seed"),
    use_wandb: bool = typer.Option(False, help="Whether to use wandb"),
    array_id: int | None = typer.Option(
        None, help="Slurm array ID for parallelization"
    ),
    num_arrays: int | None = typer.Option(
        None, help="Number of arrays for parallelization"
    ),
    sampling_yaml_path: Path = typer.Argument(
        ..., help="Path to sampling config YAML file"
    ),
) -> None:
    """Entrypoint for Protpardelle-1c sampling."""
    sample(
        sampling_yaml_path=sampling_yaml_path,
        project_name=project_name,
        motif_dir=motif_dir,
        num_samples=num_samples,
        num_mpnn_seqs=num_mpnn_seqs,
        batch_size=batch_size,
        save_shortname=save_shortname,
        seed=seed,
        use_wandb=use_wandb,
        array_id=array_id,
        num_arrays=num_arrays,
    )


if __name__ == "__main__":
    app()
