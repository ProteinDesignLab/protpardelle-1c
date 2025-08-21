"""Wrapper for drawing samples from Protpardelle-1c.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu
"""

import argparse
import shutil
import time
import uuid
from pathlib import Path

import torch.nn as nn
from torchtyping import TensorType

import protpardelle.core.diffusion as diffusion
import protpardelle.data.pdb_io
from protpardelle.utils import apply_dotdict_recursively


def default_backbone_sampling_config():
    config = argparse.Namespace(
        n_steps=500,
        s_churn=200,
        step_scale=1.2,
        sidechain_mode=False,
        noise_schedule=lambda t: diffusion.noise_schedule(t, s_max=80, s_min=0.001),
    )
    return config


def default_allatom_sampling_config():
    noise_schedule = lambda t: diffusion.noise_schedule(t, s_max=80, s_min=0.001)
    stage2 = argparse.Namespace(
        apply_cond_proportion=1.0,
        n_steps=200,
        s_churn=100,
        step_scale=1.2,
        sidechain_mode=True,
        skip_mpnn_proportion=1.0,
        noise_schedule=noise_schedule,
    )
    config = argparse.Namespace(
        n_steps=500,
        s_churn=200,
        step_scale=1.2,
        sidechain_mode=True,
        skip_mpnn_proportion=0.6,
        use_fullmpnn=False,
        use_fullmpnn_for_final=True,
        anneal_seq_resampling_rate="linear",
        noise_schedule=noise_schedule,
        stage_2=stage2,
    )
    return config


def draw_samples(
    model: nn.Module,
    seq_mask: TensorType["b n", float] | None = None,
    residue_index: TensorType["b n", float] | None = None,
    chain_index: TensorType["b n", int] | None = None,
    hotspot: str | list[str] | None = None,
    sse_cond: TensorType["b n", int] | None = None,
    adj_cond: TensorType["b n n", int] | None = None,
    motif_placements_full: list[str] | None = None,
    n_samples: int = None,
    length_ranges_per_chain: list[tuple[int, int]] = [(50, 512)],
    return_aux: bool = False,
    return_sampling_runtime: bool = False,
    return_coords_and_aux: bool = False,
    **sampling_kwargs,
):

    device = model.device
    if seq_mask is None:
        seq_mask, residue_index, chain_index = model.make_seq_mask_for_sampling(
            length_ranges_per_chain=length_ranges_per_chain, n_samples=n_samples
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
        hotspot=hotspot,
        sse_cond=sse_cond,
        adj_cond=adj_cond,
        motif_placements_full=motif_placements_full,
        return_last=False,
        return_aux=True,
        dummy_fill_mode=model.config.data.dummy_fill_mode,
        **sampling_kwargs,
    )

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

        for i in range(len(samp_coords)):
            length_bi = int(seq_mask[i].sum().item())
            protpardelle.data.pdb_io.write_coords_to_pdb(
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
            hotspot=hotspot,
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
    cropped_chain_index = [c[: seq_lens[i]] for i, c in enumerate(chain_index)]

    if return_aux:
        return aux
    else:
        if return_sampling_runtime:
            return cropped_samp_coords, cropped_chain_index, seq_mask, aux["runtime"]
        elif return_coords_and_aux:
            return cropped_samp_coords, cropped_chain_index, seq_mask, aux
        else:
            return cropped_samp_coords, cropped_chain_index, seq_mask
