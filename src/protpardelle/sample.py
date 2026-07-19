"""Entrypoint for Protpardelle-1c sampling.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import itertools
import json
import logging
import math
import os
import shutil
import subprocess
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
from protpardelle.data.pdb_io import load_feats_from_pdb, write_coords_to_pdb
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
        samp_save_name = save_dir / f"sample_{save_name}_{idx:02d}.pdb"

        if bb_only:
            coords_to_save = coords_to_save[:, (0, 1, 2, 4), :]

        write_coords_to_pdb(
            coords_to_save,
            samp_save_name,
            batched=False,
            aatype=dummy_aatype[idx],
            residue_index=trimmed_residue_index[idx],
            chain_index=trimmed_chain_index[idx],
            chain_id_mapping=samp_aux["all_chain_id_mappings"][idx],
            with_conect=not bb_only,
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
                chain_id_mapping=samp_aux["all_chain_id_mappings"][scaffold_idx],
                b_factors=all_atom_plddts[idx],
            )

    return all_samp_save_names, all_pred_save_names


def _resolve_caliby_cuda_device(caliby_cfg: dict) -> str | None:
    """Which GPU should the Caliby subprocess target (CUDA_VISIBLE_DEVICES)?

    Precedence: ``caliby_ensemble.cuda_device`` in the sampling config, then the
    ``CALIBY_CUDA_VISIBLE_DEVICES`` env var. Returns ``None`` to inherit the
    parent's ``CUDA_VISIBLE_DEVICES`` (share the main process's GPU) — fine on
    Default-compute-mode nodes but not on EXCLUSIVE_PROCESS nodes, where the
    subprocess must be pointed at a second allocated GPU.
    """

    device = caliby_cfg.get("cuda_device")
    if device is None:
        device = os.environ.get("CALIBY_CUDA_VISIBLE_DEVICES")
    return None if device is None else str(device)


def _call_caliby_subprocess_batched(
    jobs: list[dict],
    out_dir: str,
    venv_python: str,
    caliby_script: str,
    model_params_dir: str,
    checkpoint: str = "caliby",
    num_seqs: int = 1,
    temperature: float | None = None,
    cuda_visible_devices: str | None = None,
) -> list[dict]:
    """Invoke scripts/run_caliby_ensemble.py once for multiple primaries.

    Each job is a dict with keys: ``key``, ``primary_pdb``, ``ensemble_pdbs``,
    and optional ``fixed_pos_seq`` / ``fixed_pos_override_seq``. The Caliby
    model is loaded a single time inside the subprocess and each primary is
    still processed as its own ensemble batch by ``model.ensemble_sample`` —
    identical per-primary computation to running the legacy per-job subprocess
    N times, minus subprocess + caliby-import + model-load overhead.

    Returns: list of ``{"key", "seqs", "energies", "primary_seq"}`` in the same
    order as ``jobs``.
    """
    args = {
        "jobs": [
            {
                "key": j["key"],
                "primary_pdb": os.path.abspath(j["primary_pdb"]),
                "ensemble_pdbs": [os.path.abspath(p) for p in j["ensemble_pdbs"]],
                "fixed_pos_seq": j.get("fixed_pos_seq"),
                "fixed_pos_override_seq": j.get("fixed_pos_override_seq"),
            }
            for j in jobs
        ],
        "out_dir": os.path.abspath(out_dir),
        "checkpoint": checkpoint,
        "num_seqs": num_seqs,
    }
    if temperature is not None:
        args["temperature"] = float(temperature)
    os.makedirs(out_dir, exist_ok=True)
    args_path = os.path.join(out_dir, "caliby_args.json")
    with open(args_path, "w") as f:
        json.dump(args, f)
    env = os.environ.copy()
    env["MODEL_PARAMS_DIR"] = model_params_dir
    env.setdefault("PDB_MIRROR_PATH", "")
    env.setdefault("CCD_MIRROR_PATH", "")
    # Optionally pin the Caliby subprocess to a specific GPU. On nodes whose
    # GPUs are in EXCLUSIVE_PROCESS compute mode, a second process cannot open a
    # CUDA context on the same physical device the main sampling process already
    # holds one on — the subprocess dies at model load with
    # "CUDA error: CUDA-capable device(s) is/are busy or unavailable". Allocate
    # a second GPU (-G 2) and set caliby_ensemble.cuda_device (or
    # $CALIBY_CUDA_VISIBLE_DEVICES) to its index to route the subprocess there.
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    # Free the main process's cached GPU memory so the subprocess (esp. when
    # sharing a single GPU in Default compute mode) has headroom to load Caliby.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    result = subprocess.run(
        [venv_python, caliby_script, args_path],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Caliby subprocess failed (rc={result.returncode}).\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout tail:\n{result.stdout[-2000:]}"
        )
    stdout = result.stdout
    begin = stdout.find("---CALIBY_RESULT_BEGIN---")
    end = stdout.find("---CALIBY_RESULT_END---")
    if begin == -1 or end == -1:
        raise RuntimeError(f"Caliby output missing sentinels.\nstdout:\n{stdout}")
    payload = stdout[begin + len("---CALIBY_RESULT_BEGIN---"):end].strip()
    parsed = json.loads(payload)
    results_by_key = {r["key"]: r for r in parsed["results"]}
    missing = [j["key"] for j in jobs if j["key"] not in results_by_key]
    if missing:
        raise RuntimeError(f"Caliby batched output missing keys: {missing}")
    return [results_by_key[j["key"]] for j in jobs]


def _format_fixed_pos_for_caliby(
    motif_positions_0indexed: list[int],
    residue_index_for_sample: torch.Tensor,
    chain_index_for_sample: torch.Tensor,
    chain_id_mapping: dict | None,
) -> str:
    """Convert 0-indexed tensor positions to Caliby's chain+resnum format (e.g. 'A10,A25')."""
    parts = []
    if chain_id_mapping is not None:
        id_to_chain = {v: k for k, v in chain_id_mapping.items()}
    else:
        id_to_chain = None
    for pos in motif_positions_0indexed:
        resnum = int(residue_index_for_sample[pos].item())
        chain_int = int(chain_index_for_sample[pos].item())
        if id_to_chain is not None and chain_int in id_to_chain:
            chain = id_to_chain[chain_int]
        else:
            chain = chr(ord("A") + chain_int)
        parts.append(f"{chain}{resnum}")
    return ",".join(parts)


def _score_seq_with_aatype_biases(
    seq: str,
    aatype_biases: dict,
    motif_positions: set[int],
) -> tuple[float, dict]:
    """Sum aatype_biases over non-motif positions. Returns (reward, diagnostics).

    `aatype_biases` keys are one-letter codes (e.g. ``{"P": 5.0}``); missing
    entries default to 0.0. `motif_positions` is the 0-indexed set of scaffold
    positions held constant across branches (pinned via Caliby fixed_pos), so
    they're excluded from the ranking signal.
    """
    reward = 0.0
    counts: dict[str, int] = {}
    for pos, aa in enumerate(seq):
        if pos in motif_positions:
            continue
        reward += float(aatype_biases.get(aa, 0.0))
        counts[aa] = counts.get(aa, 0) + 1
    return reward, counts


def run_inference_guidance(
    model: Protpardelle,
    main_sample_pdbs: list[str],
    aux: tuple,
    sampling_config: dict,
    guidance_cfg: dict,
    caliby_cfg: dict,
    tmp_root: str,
    motif_placements_full: list[str] | None = None,
    is_unindexed_contig: bool = False,
    motif_fp: Path | None = None,
    motif_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    per_motif_save_dir: Path | None = None,
) -> list[str]:
    """Best-of-N inference-time guidance.

    For each primary, draw ``n_branches`` *independent* full 500-step samples
    (no partial_diffusion) so every branch preserves the motif geometry at the
    main-pass tolerance (~0.3 Å tip-RMSD). Single-backbone Caliby scores each
    branch with ``aatype_biases``; top-1 winner overwrites the primary PDB +
    aux. Returns winners' Caliby seqs so the caller can feed them into
    caliby_redenoise (skipping the bias-unaware 32-decoy ensemble Caliby).

    Compute: ``num_primary * n_branches`` full 500-step rollouts + one
    batched Caliby subprocess over ``num_primary * n_branches`` jobs (each
    decoding ``num_seqs_per_branch`` sequences).

    If ``motif_fp`` and ``per_motif_save_dir`` are supplied, motif_rotated_NN.pdb
    is also rewritten to the winner's random rotation so it visually matches
    the final structure.
    """
    n_branches = int(guidance_cfg["n_branches"])
    k_keep = int(guidance_cfg.get("k_keep", 1))
    # How many Caliby sequences to draw per branch backbone. Caliby's
    # per-job cost is dominated by structure encoding, so asking for many seqs
    # is much cheaper than running more branches. With this set, we treat each
    # (branch, seq) pair as a candidate and pick the absolute top-1 by reward.
    num_seqs_per_branch = int(guidance_cfg.get("num_seqs_per_branch", 1))
    # Caliby sampling temperature. Higher T = more diverse seqs (more likely
    # to include rare residues like P). Default is whatever caliby uses
    # internally (currently ~0.1, very greedy).
    caliby_temperature = guidance_cfg.get("caliby_temperature")
    aatype_biases = dict(guidance_cfg.get("reward", {}).get("aatype_biases", {}) or {})

    if k_keep != 1:
        raise NotImplementedError(
            f"inference_guidance.k_keep={k_keep} not supported; only k_keep=1 implemented"
        )

    venv_python = caliby_cfg["venv_python"]
    caliby_script = caliby_cfg.get("caliby_script") or str(
        Path(__file__).resolve().parents[2] / "scripts" / "run_caliby_ensemble.py"
    )
    model_params_dir = caliby_cfg["model_params_dir"]
    checkpoint = caliby_cfg.get("checkpoint", "caliby")
    caliby_cuda_device = _resolve_caliby_cuda_device(caliby_cfg)

    trimmed_coords, trimmed_residue_index, trimmed_chain_index, seq_mask, samp_aux, _ = aux
    num_main = len(main_sample_pdbs)
    device = next(model.parameters()).device

    logger.info(
        "[Guidance] best-of-N: n_branches=%d, num_seqs_per_branch=%d, "
        "caliby_temperature=%s, aatype_biases=%s",
        n_branches, num_seqs_per_branch, caliby_temperature, aatype_biases,
    )

    # Phase 1: per primary, fork into n_branches and queue Caliby jobs. We
    # accumulate jobs across all primaries so a single batched Caliby subprocess
    # handles them all — avoids re-loading the Caliby model num_main times.
    caliby_jobs: list[dict] = []
    branch_state: dict[str, dict] = {}

    for i, main_pdb in enumerate(main_sample_pdbs):
        logger.info(
            "[Guidance] primary %d/%d: generating %d full-sample branches",
            i + 1, num_main, n_branches,
        )
        primary_dir = Path(tmp_root) / f"primary_{i:02d}"
        primary_dir.mkdir(parents=True, exist_ok=True)
        branch_dir = primary_dir / "branches"
        branch_dir.mkdir(exist_ok=True)

        # Per-sample sampling config. No partial_diffusion — every branch is
        # an independent fresh 500-step sampling, so it preserves motif geometry
        # at the main-pass tolerance. Disable downstream ensemble Caliby to
        # prevent recursion when draw_samples re-reads the config.
        branch_cfg = deepcopy(sampling_config)
        branch_cfg["sampling"]["partial_diffusion"]["enabled"] = False
        branch_cfg["sampling"]["allatom_cfg"]["caliby_ensemble"]["enabled"] = False

        # Replicate primary i's length/index info across n_branches so all
        # branches share the same residue numbering as the main sample.
        ri = trimmed_residue_index[i]
        ci = trimmed_chain_index[i]
        L_i = ri.shape[0]
        seq_mask_branches = torch.ones((n_branches, L_i), dtype=torch.float, device=device)
        residue_index_branches = ri.to(device).unsqueeze(0).repeat(n_branches, 1)
        chain_index_branches = ci.to(device).unsqueeze(0).repeat(n_branches, 1)

        branch_coords, branch_res_idx, branch_ch_idx, branch_seq_mask, branch_aux = draw_samples(
            model,
            num_samples=n_branches,
            seq_mask=seq_mask_branches,
            residue_index=residue_index_branches,
            chain_index=chain_index_branches,
            length_ranges_per_chain=None,
            return_coords_and_aux=True,
            motif_placements_full=motif_placements_full,
            **branch_cfg["sampling"],
        )

        branch_chain_id_mappings = branch_aux.get(
            "chain_id_mapping",
            branch_aux.get("all_chain_id_mappings", [None] * n_branches),
        )
        branch_atom_mask_list = branch_aux.get("atom_mask")
        # With full sampling each branch has its OWN linear_sum_assignment
        # matching; use those for fixed_pos pinning rather than the main
        # sample's, since branches' motif positions can differ from main's.
        branch_motif_idx_list = branch_aux.get("motif_idx") or [[] for _ in range(n_branches)]
        branch_motif_aatype_list = branch_aux.get("motif_aatype")
        branch_motif_random_rots = branch_aux.get("motif_random_rots")
        branch_motif_centering_delta = branch_aux.get("motif_centering_delta")
        # motif_all_atom is each branch's rotated motif coords; needed so
        # caliby_redenoise gets the winner's motif frame as motif_all_atom_stage1.
        branch_motif_all_atom = branch_aux.get("motif_all_atom")

        for k in range(n_branches):
            mp_raw = branch_motif_idx_list[k] if k < len(branch_motif_idx_list) else None
            mp = [int(x) for x in mp_raw] if mp_raw is not None else []

            # Build fixed_pos_seq + fixed_pos_override_seq so Caliby holds motif
            # AAs constant during sequence design.
            fixed_pos_seq = None
            fixed_pos_override_seq = None
            if len(mp) > 0:
                chain_id_mapping_k = (
                    branch_chain_id_mappings[k]
                    if k < len(branch_chain_id_mappings) else None
                )
                fixed_pos_seq = _format_fixed_pos_for_caliby(
                    list(mp),
                    branch_res_idx[k],
                    branch_ch_idx[k],
                    chain_id_mapping_k,
                )
                if (
                    branch_motif_aatype_list is not None
                    and k < len(branch_motif_aatype_list)
                ):
                    aa1_list = residue_constants.restypes
                    override_parts = []
                    for rel_pos, aa_idx in zip(mp, branch_motif_aatype_list[k]):
                        rel_pos_i = int(rel_pos)
                        resnum = int(branch_res_idx[k][rel_pos_i].item())
                        chain_int = int(branch_ch_idx[k][rel_pos_i].item())
                        chain = chr(ord("A") + chain_int)
                        aa_i = int(aa_idx.item()) if isinstance(aa_idx, torch.Tensor) else int(aa_idx)
                        if 0 <= aa_i < len(aa1_list):
                            override_parts.append(f"{chain}{resnum}:{aa1_list[aa_i]}")
                    if override_parts:
                        fixed_pos_override_seq = ",".join(override_parts)

            pdb_path = branch_dir / f"branch_{k:02d}.pdb"
            chain_id_mapping_k = (
                branch_chain_id_mappings[k]
                if k < len(branch_chain_id_mappings) else None
            )
            branch_atom_mask_k = None
            if branch_atom_mask_list is not None and k < len(branch_atom_mask_list):
                branch_atom_mask_k = branch_atom_mask_list[k]
            write_coords_to_pdb(
                branch_coords[k],
                pdb_path,
                batched=False,
                aatype=branch_aux["s"][k],
                atom_mask=branch_atom_mask_k,
                residue_index=branch_res_idx[k],
                chain_index=branch_ch_idx[k],
                chain_id_mapping=chain_id_mapping_k,
                with_conect=True,
            )

            job_key = f"primary_{i:02d}_branch_{k:02d}"
            caliby_jobs.append({
                "key": job_key,
                "primary_pdb": str(pdb_path),
                # Single-backbone Caliby: pass only the primary, no ensemble
                # averaging. Caliby's clean_pdbs handles this via decoys=[].
                "ensemble_pdbs": [],
                "fixed_pos_seq": fixed_pos_seq,
                "fixed_pos_override_seq": fixed_pos_override_seq,
            })
            # Capture this branch's motif rotation + centering so we can rewrite
            # motif_rotated_NN.pdb for the winner. Each full-sample branch has
            # its own random rotation (no shared frame between branches).
            branch_rot = None
            if branch_motif_random_rots is not None and k < len(branch_motif_random_rots):
                branch_rot = branch_motif_random_rots[k]
            branch_motif_all_atom_k = None
            if branch_motif_all_atom is not None and k < len(branch_motif_all_atom):
                branch_motif_all_atom_k = branch_motif_all_atom[k]
            branch_state[job_key] = {
                "primary_i": i,
                "branch_k": k,
                "pdb_path": str(pdb_path),
                "motif_idx": list(mp),
                "motif_aatype": (
                    branch_motif_aatype_list[k]
                    if branch_motif_aatype_list is not None and k < len(branch_motif_aatype_list)
                    else None
                ),
                "motif_all_atom": branch_motif_all_atom_k,
                "coords": branch_coords[k],
                "res_idx": branch_res_idx[k],
                "ch_idx": branch_ch_idx[k],
                "aatype": branch_aux["s"][k],
                "atom_mask": branch_atom_mask_k,
                "chain_id_mapping": chain_id_mapping_k,
                "motif_random_rot": branch_rot,
                "motif_centering_delta": branch_motif_centering_delta,
            }

    # Phase 2: single batched Caliby subprocess for all (primary, branch) jobs.
    logger.info(
        "[Guidance] invoking batched Caliby for %d branches (%d primaries × %d each)",
        len(caliby_jobs), num_main, n_branches,
    )
    batched_results = _call_caliby_subprocess_batched(
        jobs=caliby_jobs,
        out_dir=str(Path(tmp_root) / "caliby_batched_out"),
        venv_python=venv_python,
        caliby_script=caliby_script,
        model_params_dir=model_params_dir,
        checkpoint=checkpoint,
        num_seqs=num_seqs_per_branch,
        temperature=caliby_temperature,
        cuda_visible_devices=caliby_cuda_device,
    )

    # Phase 3: for each (branch, caliby seq) pair, compute reward. The branch's
    # contribution is the highest-reward seq across its num_seqs_per_branch
    # candidates; selection then picks the top-1 branch.
    rewards_by_primary: dict[int, list[tuple[int, float, str, dict]]] = {}
    for job, res in zip(caliby_jobs, batched_results):
        info = branch_state[job["key"]]
        motif_positions = set(info["motif_idx"])
        # Score every Caliby sample drawn for this branch; keep the best.
        best_branch_reward = -float("inf")
        best_branch_seq = None
        best_branch_counts = None
        for seq in res["seqs"]:
            reward, counts = _score_seq_with_aatype_biases(seq, aatype_biases, motif_positions)
            if reward > best_branch_reward:
                best_branch_reward = reward
                best_branch_seq = seq
                best_branch_counts = counts
        rewards_by_primary.setdefault(info["primary_i"], []).append(
            (info["branch_k"], best_branch_reward, best_branch_seq, best_branch_counts)
        )
        biased_aa_counts = {aa: best_branch_counts.get(aa, 0) for aa in aatype_biases}
        logger.info(
            "[Guidance] primary %d branch %d: reward=%.2f (best of %d), biased_AA_non_motif_counts=%s, seq_len=%d",
            info["primary_i"], info["branch_k"], best_branch_reward, len(res["seqs"]),
            biased_aa_counts, len(best_branch_seq),
        )

    # Phase 4: top-1 selection per primary; overwrite primary PDB + aux.
    # Also collect winner sequences for the caller to feed into caliby_redenoise.
    winner_seqs: list[str | None] = [None] * num_main
    for i in range(num_main):
        ranked = sorted(
            rewards_by_primary.get(i, []), key=lambda t: t[1], reverse=True
        )
        if not ranked:
            logger.warning("[Guidance] primary %d: no branch scored, leaving original", i)
            continue
        best_k, best_reward, best_seq, _best_counts = ranked[0]
        scores_summary = ", ".join(f"b{k}:{r:.2f}" for k, r, _, _ in ranked)
        logger.info(
            "[Guidance] primary %d: WINNER branch %d (reward=%.2f, seq=%s); ranked=[%s]",
            i, best_k, best_reward, best_seq, scores_summary,
        )
        winner_seqs[i] = best_seq
        info = branch_state[f"primary_{i:02d}_branch_{best_k:02d}"]

        # Critical: write the winner PDB with the CALIBY seq as aatype, not the
        # branch's ProteinMPNN-designed seq. caliby_redenoise's underlying
        # model.sample reads `pd_motif_aatype` from the input PDB and IGNORES
        # `gt_aatype` in partial_diffusion mode (models.py:1735). So unless the
        # PDB on disk carries the Caliby seq, the proline-biased sequence is
        # silently dropped on the final redenoise pass.
        def _to_cpu(t):
            return t.cpu() if isinstance(t, torch.Tensor) else t

        winner_aatype = seq_to_aatype(best_seq.replace(":", "")).to("cpu")
        winner_coords = _to_cpu(info["coords"])
        winner_atom_mask = _to_cpu(info["atom_mask"]) if info["atom_mask"] is not None else None
        # The branch's atom_mask was built around its ProteinMPNN seq, not the
        # Caliby seq, so recompute it from the Caliby aatype to keep atom
        # presence consistent with the written sequence.
        from protpardelle.data.atom import atom37_mask_from_aatype as _a37m
        seq_mask_one = torch.ones_like(winner_aatype, dtype=torch.float)
        recomputed_mask = _a37m(winner_aatype.unsqueeze(0), seq_mask_one.unsqueeze(0))[0]
        write_coords_to_pdb(
            winner_coords,
            main_sample_pdbs[i],
            batched=False,
            aatype=winner_aatype,
            atom_mask=recomputed_mask,
            residue_index=_to_cpu(info["res_idx"]),
            chain_index=_to_cpu(info["ch_idx"]),
            chain_id_mapping=info["chain_id_mapping"],
            with_conect=True,
        )

        # In-memory aux: mutate the lists in place so downstream save_samples,
        # scaffold_info.csv, and run_caliby_ensemble_design see the winner.
        trimmed_coords[i] = winner_coords
        trimmed_residue_index[i] = _to_cpu(info["res_idx"])
        trimmed_chain_index[i] = _to_cpu(info["ch_idx"])
        if samp_aux.get("s") is not None and i < len(samp_aux["s"]):
            samp_aux["s"][i] = winner_aatype
        if samp_aux.get("atom_mask") is not None and i < len(samp_aux["atom_mask"]):
            samp_aux["atom_mask"][i] = recomputed_mask
        # Each full-sample branch has its OWN linear_sum_assignment matching
        # and its OWN random rotation of the motif, so we must overwrite the
        # main sample's motif_idx/aatype/rotation with the winner's, or
        # downstream scaffold_info + motif_rotated_NN.pdb will mismatch.
        if samp_aux.get("motif_idx") is not None and i < len(samp_aux["motif_idx"]):
            samp_aux["motif_idx"][i] = list(info["motif_idx"])
        if (
            info["motif_aatype"] is not None
            and samp_aux.get("motif_aatype") is not None
            and i < len(samp_aux["motif_aatype"])
        ):
            samp_aux["motif_aatype"][i] = _to_cpu(info["motif_aatype"])
        if (
            info["motif_random_rot"] is not None
            and samp_aux.get("motif_random_rots") is not None
            and i < len(samp_aux["motif_random_rots"])
        ):
            samp_aux["motif_random_rots"][i] = _to_cpu(info["motif_random_rot"])
        # caliby_redenoise reads samp_aux["motif_coords"] as motif_all_atom_stage1.
        # If we leave it as the main sample's coords while the on-disk PDB is
        # the winner's structure, the redenoise rewinds with mismatched motif
        # frames and drifts tips by ~5 Å. Overwriting keeps the redenoise on
        # the winner's frame.
        if (
            info["motif_all_atom"] is not None
            and samp_aux.get("motif_coords") is not None
            and i < len(samp_aux["motif_coords"])
        ):
            samp_aux["motif_coords"][i] = _to_cpu(info["motif_all_atom"])
        if (
            info["chain_id_mapping"] is not None
            and samp_aux.get("all_chain_id_mappings") is not None
            and i < len(samp_aux["all_chain_id_mappings"])
        ):
            samp_aux["all_chain_id_mappings"][i] = info["chain_id_mapping"]

        # Rewrite motif_rotated_NN.pdb with the winner's rotation+centering so
        # PyMOL/visual overlay matches the final structure. The main loop wrote
        # an early version using the main sample's rotation; overwrite here.
        if (
            motif_fp is not None
            and per_motif_save_dir is not None
            and info["motif_random_rot"] is not None
            and info["motif_centering_delta"] is not None
        ):
            cd_np = _to_cpu(info["motif_centering_delta"])
            if hasattr(cd_np, "numpy"):
                cd_np = cd_np.numpy()
            # motif_centering_delta is broadcast across batch; extract the 3-vec
            cd_vec = np.asarray(cd_np).reshape(-1, 3)[0]
            rot_np = _to_cpu(info["motif_random_rot"])
            if hasattr(rot_np, "numpy"):
                rot_np = rot_np.numpy()
            cond_strategy = sampling_config["sampling"]["conditional_cfg"][
                "crop_conditional_guidance"
            ].get("strategy")
            _write_transformed_motif_pdbs(
                Path(motif_fp),
                Path(per_motif_save_dir),
                i,
                centering_delta=cd_vec,
                rotation=rot_np,
                translation=np.asarray(motif_translation, dtype=float),
                strategy=cond_strategy,
            )

    return [s if s is not None else "" for s in winner_seqs]


def run_caliby_ensemble_design(
    model: Protpardelle,
    main_sample_pdbs: list[str],
    aux: tuple,
    motif_fp: Path | list | None,
    motif_placements_full: list[str] | None,
    sampling_config: dict,
    caliby_cfg: dict,
    tmp_root: str,
    is_unindexed_contig: bool = False,
) -> list[str]:
    """Two-pass orchestration: generate ensembles via partial_diffusion, then call Caliby.

    Returns: list of designed sequences (one per main sample).
    """
    trimmed_coords, trimmed_residue_index, trimmed_chain_index, seq_mask, samp_aux, _ = aux
    num_main = len(main_sample_pdbs)
    ensemble_size = int(caliby_cfg.get("ensemble_size", 32))
    rewind_steps = int(caliby_cfg.get("rewind_steps", 150))
    checkpoint = caliby_cfg.get("checkpoint", "caliby")
    beam_strategy = caliby_cfg.get("beam_strategy", "minimpnn")
    venv_python = caliby_cfg["venv_python"]
    caliby_script = caliby_cfg.get("caliby_script") or str(
        Path(__file__).resolve().parents[2] / "scripts" / "run_caliby_ensemble.py"
    )
    model_params_dir = caliby_cfg["model_params_dir"]
    caliby_cuda_device = _resolve_caliby_cuda_device(caliby_cfg)

    if beam_strategy not in ("minimpnn", "uniform_random", "caliby_x0", "backbone_only"):
        raise ValueError(f"Unknown beam_strategy: {beam_strategy}")
    if beam_strategy == "caliby_x0":
        raise NotImplementedError(
            "beam_strategy=caliby_x0 not yet implemented: would require ~150 Caliby subprocess "
            "calls per decoy per ensemble member (very expensive). Use a coarser "
            "caliby_x0_freq if this strategy becomes necessary."
        )

    # Optionally load a separate backbone-only model for the ensemble-generation pass.
    bb_model = None
    if beam_strategy == "backbone_only":
        bb_name = caliby_cfg.get("backbone_model_name", "cc95")
        bb_epoch = int(caliby_cfg.get("backbone_model_epoch", 3490))
        bb_cfg_path = str(PROTPARDELLE_MODEL_PARAMS / "configs" / f"{bb_name}.yaml")
        bb_ckpt_path = str(PROTPARDELLE_MODEL_PARAMS / "weights" / f"{bb_name}_epoch{bb_epoch}.pth")
        logger.info("[Caliby] loading backbone-only model for ensemble: %s epoch %d", bb_name, bb_epoch)
        bb_model = load_model(bb_cfg_path, bb_ckpt_path)

    motif_idx = samp_aux.get("motif_idx")
    motif_aatype = samp_aux.get("motif_aatype")
    # Per-motif-residue atom mask; rows that are all-zero correspond to motif
    # residues unmodeled in the source PDB (gaps padded to the nominal contig
    # length). Those positions are model-generated filler with no known native
    # identity, so they must be designed by Caliby rather than pinned.
    motif_atom_mask = samp_aux.get("motif_atom_mask")

    # Phase 1: generate one ensemble per main sample (still sequential because each
    # ensemble is its own batched partial_diffusion call). Accumulate jobs to send
    # to Caliby in a single batched subprocess call.
    caliby_jobs: list[dict] = []

    for i, main_pdb in enumerate(main_sample_pdbs):
        logger.info("[Caliby] main sample %d/%d: generating %d-member ensemble (strategy=%s)",
                    i + 1, num_main, ensemble_size, beam_strategy)
        tmp_dir = Path(tmp_root) / f"caliby_{i:02d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ens_dir = tmp_dir / "ensemble"
        ens_dir.mkdir(exist_ok=True)

        # Build a per-sample sampling config for the partial_diffusion pass.
        pd_cfg = deepcopy(sampling_config)
        pd_cfg["sampling"]["partial_diffusion"]["enabled"] = True
        pd_cfg["sampling"]["partial_diffusion"]["n_steps"] = rewind_steps
        # draw_samples expects a single path for feats loading; model.sample expands
        # it to batch_size copies internally.
        pd_cfg["sampling"]["partial_diffusion"]["pdb_file_path"] = main_pdb
        pd_cfg["sampling"]["allatom_cfg"]["caliby_ensemble"]["enabled"] = False  # avoid recursion

        ds_model = bb_model if beam_strategy == "backbone_only" else model
        ds_kwargs = dict(
            num_samples=ensemble_size,
            length_ranges_per_chain=None,
            return_coords_and_aux=True,
        )

        if beam_strategy == "uniform_random":
            # Override MPNN resampling: set gt_aatype to a batch of uniform-random
            # sequences (one per ensemble member), effectively freezing MPNN to a
            # random prior throughout the 350->500 pass.
            rng_seed = caliby_cfg.get("uniform_random_seed")
            if rng_seed is not None:
                torch.manual_seed(int(rng_seed) + i)
            # Length comes from the main sample.
            seq_len = len(trimmed_residue_index[i])
            n_aa = len(residue_constants.restypes)  # 20 canonical
            random_aatypes = torch.randint(
                low=0, high=n_aa, size=(ensemble_size, seq_len),
                dtype=torch.long, device=ds_model.device,
            )
            ds_kwargs["gt_aatype"] = random_aatypes
            # With gt_aatype set, don't run MPNN during the pass.
            pd_cfg["sampling"]["allatom_cfg"]["skip_mpnn_proportion"] = 1.0
            pd_cfg["sampling"]["allatom_cfg"]["use_fullmpnn_for_final"] = False

        if beam_strategy == "backbone_only":
            # cc95/bb81/cc58 are task=backbone — they don't run MPNN at all and
            # only emit backbone coords. Turn off allatom-specific knobs.
            pd_cfg["sampling"]["allatom_cfg"]["sidechain_mode"] = False
            pd_cfg["sampling"]["allatom_cfg"]["jump_steps"] = False
            pd_cfg["sampling"]["allatom_cfg"]["uniform_steps"] = True
            pd_cfg["sampling"]["allatom_cfg"]["use_fullmpnn_for_final"] = False
            pd_cfg["sampling"]["allatom_cfg"]["skip_mpnn_proportion"] = 1.0
            # cc95 uses conditioning_style="concat_hotspot"; cc58 is plain "concat".
            # Our current motif conditioning is acidic-tip (requires condmask).
            # Since bb models don't have condmask, disable all motif conditioning
            # paths for the bb pass — accept that the motif may drift slightly.
            pd_cfg["sampling"]["conditional_cfg"]["enabled"] = False
            pd_cfg["sampling"]["conditional_cfg"]["crop_conditional_guidance"]["enabled"] = False
            pd_cfg["sampling"]["conditional_cfg"]["reconstruction_guidance"]["enabled"] = False
            pd_cfg["sampling"]["conditional_cfg"]["replacement_guidance"]["enabled"] = False

        # Run partial_diffusion on a batch of ensemble_size copies.
        ens_trimmed_coords, ens_trimmed_res_idx, ens_trimmed_ch_idx, ens_seq_mask, ens_samp_aux = draw_samples(
            ds_model,
            **ds_kwargs,
            **pd_cfg["sampling"],
        )

        # Write ensemble PDBs
        # draw_samples returns samp_aux with "chain_id_mapping" (list per sample).
        ens_chain_id_mappings = ens_samp_aux.get(
            "chain_id_mapping",
            ens_samp_aux.get("all_chain_id_mappings", [None] * ensemble_size),
        )
        ensemble_pdb_paths = []
        for k in range(ensemble_size):
            pdb_path = ens_dir / f"decoy_{k:02d}.pdb"
            write_coords_to_pdb(
                ens_trimmed_coords[k],
                pdb_path,
                batched=False,
                aatype=ens_samp_aux["s"][k],
                residue_index=ens_trimmed_res_idx[k],
                chain_index=ens_trimmed_ch_idx[k],
                chain_id_mapping=ens_chain_id_mappings[k] if k < len(ens_chain_id_mappings) else None,
                with_conect=True,
            )
            ensemble_pdb_paths.append(str(pdb_path))

        # Build fixed_pos_seq for Caliby so motif positions are locked to motif aatypes
        fixed_pos_seq = None
        fixed_pos_override_seq = None
        if motif_idx is not None and motif_aatype is not None and len(motif_idx) > i:
            m_pos = motif_idx[i]
            m_aatype = motif_aatype[i]
            # Drop padded gap positions (all-zero atom mask): they have no native
            # identity and are designed, not fixed.
            if motif_atom_mask is not None and i < len(motif_atom_mask):
                present = (motif_atom_mask[i].sum(-1) > 0).cpu().tolist()
                if len(present) == len(m_pos):
                    kept = [k for k, p in enumerate(present) if p]
                    m_pos = [m_pos[k] for k in kept]
                    if isinstance(m_aatype, torch.Tensor):
                        m_aatype = m_aatype[torch.tensor(kept, dtype=torch.long)]
                    else:
                        m_aatype = [m_aatype[k] for k in kept]
            if m_pos is not None and len(m_pos) > 0:
                fixed_pos_seq = _format_fixed_pos_for_caliby(
                    m_pos,
                    trimmed_residue_index[i],
                    trimmed_chain_index[i],
                    samp_aux["all_chain_id_mappings"][i],
                )
                # Also override sequence at those positions to the motif aatypes
                # since the main sample's MPNN seq may already have them correct,
                # but primary cleaned PDB could have drifted.
                override_parts = []
                aa1_list = residue_constants.restypes
                for rel_pos, aa_idx in zip(m_pos, m_aatype):
                    resnum = int(trimmed_residue_index[i][rel_pos].item())
                    chain_int = int(trimmed_chain_index[i][rel_pos].item())
                    chain = chr(ord("A") + chain_int)
                    aa_i = int(aa_idx.item()) if isinstance(aa_idx, torch.Tensor) else int(aa_idx)
                    if 0 <= aa_i < len(aa1_list):
                        override_parts.append(f"{chain}{resnum}:{aa1_list[aa_i]}")
                if override_parts:
                    fixed_pos_override_seq = ",".join(override_parts)

        # Queue this main sample as a job for the batched Caliby call below.
        # ensemble_sample processes each primary as its own batch internally,
        # so per-primary computation is identical to issuing N separate
        # subprocesses — we just collapse N model loads into 1.
        caliby_jobs.append({
            "key": f"main_{i:02d}",
            "primary_pdb": main_pdb,
            "ensemble_pdbs": ensemble_pdb_paths,
            "fixed_pos_seq": fixed_pos_seq,
            "fixed_pos_override_seq": fixed_pos_override_seq,
        })
        logger.info("[Caliby] main sample %d/%d: queued (n_decoys=%d, fixed_pos=%s)",
                    i + 1, num_main, len(ensemble_pdb_paths),
                    fixed_pos_seq if fixed_pos_seq else "none")

    # Phase 2: single batched Caliby subprocess for all main samples.
    logger.info("[Caliby] invoking batched Caliby ensemble design for %d main samples", len(caliby_jobs))
    batched_results = _call_caliby_subprocess_batched(
        jobs=caliby_jobs,
        out_dir=str(Path(tmp_root) / "caliby_batched_out"),
        venv_python=venv_python,
        caliby_script=caliby_script,
        model_params_dir=model_params_dir,
        checkpoint=checkpoint,
        num_seqs=1,
        cuda_visible_devices=caliby_cuda_device,
    )

    caliby_seqs: list[str] = []
    for i, res in enumerate(batched_results):
        seq = res["seqs"][0]
        energy = res["energies"][0]
        logger.info("[Caliby] main sample %d: designed seq len=%d, U=%.2f",
                    i + 1, len(seq), energy)
        caliby_seqs.append(seq)

    return caliby_seqs


def caliby_redenoise(
    model: Protpardelle,
    main_sample_pdbs: list[str],
    caliby_seqs: list[str],
    aux: tuple,
    sampling_config: dict,
    caliby_cfg: dict,
    motif_placements_full: list[str] | None = None,
) -> None:
    """Third pass: re-denoise 350->500 with gt_aatype fixed to Caliby's sequence.

    Uses partial_diffusion (which re-noises the main sample PDB up to step 350 via
    the superposition scheme, using the provided aatype to determine atom tracks),
    then denoises down to step 500 with gt_aatype fixed. Overwrites the main sample
    PDBs on disk and updates aux in place so save_samples / downstream code see
    Caliby-consistent coords + sequences.
    """
    trimmed_coords, trimmed_residue_index, trimmed_chain_index, seq_mask, samp_aux, _ = aux
    rewind_steps = int(caliby_cfg.get("rewind_steps", 150))
    device = model.device
    num_main = len(main_sample_pdbs)

    # Convert Caliby seqs to aatype tensors, padded to max seq length in batch
    caliby_aatypes = [seq_to_aatype(s.replace(":", "")) for s in caliby_seqs]
    max_len = max(t.shape[0] for t in caliby_aatypes)
    gt_aatype_batched = torch.zeros((num_main, max_len), dtype=torch.long, device=device)
    for i, t in enumerate(caliby_aatypes):
        gt_aatype_batched[i, : t.shape[0]] = t.to(device)

    # Build per-sample config matching sample.py's stage2 pattern
    pd_cfg = deepcopy(sampling_config)
    pd_cfg["sampling"]["partial_diffusion"]["enabled"] = True
    pd_cfg["sampling"]["partial_diffusion"]["n_steps"] = rewind_steps
    pd_cfg["sampling"]["partial_diffusion"]["pdb_file_path"] = [
        os.path.abspath(p) for p in main_sample_pdbs
    ]
    # With gt_aatype, we don't need MPNN resampling during the final 350->500 pass
    pd_cfg["sampling"]["allatom_cfg"]["skip_mpnn_proportion"] = 1.0
    pd_cfg["sampling"]["allatom_cfg"]["jump_steps"] = False
    pd_cfg["sampling"]["allatom_cfg"]["uniform_steps"] = True
    pd_cfg["sampling"]["allatom_cfg"]["use_fullmpnn_for_final"] = False
    pd_cfg["sampling"]["allatom_cfg"]["caliby_ensemble"]["enabled"] = False  # avoid recursion
    # Keep motif conditioning enabled so tip atoms stay pinned. We pass
    # motif_all_atom_stage1 / motif_idx_stage1 (from the original sample) so the
    # model skips the load_feats_from_pdb(pd.pdb_file_path) call that chokes on
    # a list of paths (see stage2 pattern).

    allatom_cfg = apply_dotdict_recursively(pd_cfg["sampling"]["allatom_cfg"])
    stage2_cfg = apply_dotdict_recursively(pd_cfg["sampling"]["stage2_cfg"])
    if allatom_cfg is not None and "caliby_ensemble" in allatom_cfg:
        allatom_cfg.pop("caliby_ensemble")
    sampling_kwargs = deepcopy(pd_cfg["sampling"])
    sampling_kwargs.pop("allatom_cfg")
    sampling_kwargs.pop("stage2_cfg")
    sampling_kwargs.update(allatom_cfg)

    # partial_diffusion loads residue_index / chain_index from the PDB per sample;
    # but model.sample expects a single batched tensor. We pass residue_index from
    # the first main PDB — they all match the original scaffold since the PDBs come
    # from the same motif/contig run.
    from protpardelle.data.pdb_io import load_feats_from_pdb as _load_feats
    pd_feats_all = [_load_feats(p, include_pos_feats=True)[0] for p in main_sample_pdbs]
    max_n = max(f["residue_index"].shape[0] for f in pd_feats_all)
    residue_index_b = torch.zeros((num_main, max_n), dtype=torch.long, device=device)
    chain_index_b = torch.zeros((num_main, max_n), dtype=torch.long, device=device)
    seq_mask_b = torch.zeros((num_main, max_n), dtype=torch.float, device=device)
    for i, f in enumerate(pd_feats_all):
        L = f["residue_index"].shape[0]
        residue_index_b[i, :L] = f["residue_index"].to(device)
        chain_index_b[i, :L] = f["chain_index"].to(device)
        seq_mask_b[i, :L] = 1.0

    # Cached motif state from the original sample — passed to bypass the
    # load_feats_from_pdb(list-of-paths) bug in model.sample (line ~1305).
    motif_all_atom_cache = samp_aux.get("motif_coords")
    motif_idx_cache = samp_aux.get("motif_idx")

    logger.info("[Caliby] re-denoising pass: partial_diffusion with gt_aatype for %d samples", num_main)
    redenoise_aux = model.sample(
        seq_mask=seq_mask_b,
        residue_index=residue_index_b,
        chain_index=chain_index_b,
        gt_aatype=gt_aatype_batched,
        # For indexed runs we pass motif_placements_full so model.sample builds
        # motif_aatype as a per-batch list (required by apply_crop_cond_strategy).
        # For unindexed runs this should be None.
        motif_placements_full=motif_placements_full,
        motif_all_atom_stage1=motif_all_atom_cache,
        motif_idx_stage1=motif_idx_cache,
        stage2=True,
        return_last=False,
        return_aux=True,
        dummy_fill_mode=model.config.data.dummy_fill_mode,
        **sampling_kwargs,
    )
    new_coords = redenoise_aux["x"]
    new_s = redenoise_aux["s"]
    new_atom_mask = redenoise_aux["atom_mask"]

    # Overwrite on-disk PDBs and update in-memory aux so save_samples + scaffold_info are consistent.
    from protpardelle.data.atom import atom37_mask_from_aatype as _a37m
    for i, pdb_path in enumerate(main_sample_pdbs):
        L = int(seq_mask_b[i].sum().item())
        coords_i = new_coords[i, :L]
        aatype_i = new_s[i, :L]
        mask_i = new_atom_mask[i, :L]
        write_coords_to_pdb(
            coords_i.cpu(),
            Path(pdb_path),
            batched=False,
            aatype=aatype_i.cpu(),
            atom_mask=mask_i.cpu(),
            residue_index=residue_index_b[i, :L].cpu(),
            chain_index=chain_index_b[i, :L].cpu(),
            chain_id_mapping=samp_aux["all_chain_id_mappings"][i],
            with_conect=True,
        )
        # Update in-memory aux for downstream
        samp_aux["s"][i] = aatype_i.cpu()
        trimmed_coords[i] = coords_i.cpu()


def draw_samples(
    model: Protpardelle,
    seq_mask: TensorType["b n", float] | None = None,
    residue_index: TensorType["b n", float] | None = None,
    chain_index: TensorType["b n", int] | None = None,
    hotspots: str | list[str] | None = None,
    sse_cond: TensorType["b n", int] | None = None,
    adj_cond: TensorType["b n n", int] | None = None,
    motif_sse_label: str | None = None,
    motif_sse_flank: int | None = None,
    binder_sse_prop: list[float] | tuple[float, float, float] | None = None,
    motif_placements_full: list[str] | None = None,
    num_samples: int | None = None,
    length_ranges_per_chain: list[tuple[int, int]] | None = None,
    return_aux: bool = False,
    return_sampling_runtime: bool = False,
    return_coords_and_aux: bool = False,
    xt_0_path: Path | None = None,
    xt_1_path: Path | None = None,
    **sampling_kwargs,
):

    device = model.device
    if seq_mask is None and length_ranges_per_chain is not None:
        seq_mask, residue_index, chain_index = model.make_seq_mask_for_sampling(
            length_ranges_per_chain=length_ranges_per_chain, num_samples=num_samples
        )
    chain_id_mapping = [None] * num_samples
    if sampling_kwargs["partial_diffusion"]["enabled"]:
        pd_feats, pd_hetero_obj = load_feats_from_pdb(sampling_kwargs["partial_diffusion"]["pdb_file_path"], include_pos_feats=True)
        # update residue index based on partial diffusion input PDB (chain breaks + multiple chains)
        residue_index = torch.tile(pd_feats["residue_index"][None], (num_samples, 1)).to(device)
        residue_index_orig = torch.tile(pd_feats["residue_index_orig"][None], (num_samples, 1)).to(device)
        chain_index = torch.tile(pd_feats["chain_index"][None], (num_samples, 1)).to(device)
        chain_id_mapping = [pd_feats["chain_id_mapping"]] * num_samples
        seq_mask = torch.ones_like(residue_index).to(device)

    start = time.time()

    allatom_cfg = apply_dotdict_recursively(sampling_kwargs["allatom_cfg"])
    stage2_cfg = apply_dotdict_recursively(sampling_kwargs["stage2_cfg"])
    sampling_kwargs.pop("allatom_cfg")
    sampling_kwargs.pop("stage2_cfg")
    # caliby_ensemble is orchestrated in sample.py; strip before passing to model.sample()
    if allatom_cfg is not None and "caliby_ensemble" in allatom_cfg:
        allatom_cfg.pop("caliby_ensemble")
    sampling_kwargs.update(allatom_cfg)

    aux = model.sample(
        seq_mask=seq_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        hotspots=hotspots,
        sse_cond=sse_cond,
        adj_cond=adj_cond,
        motif_sse_label=motif_sse_label,
        motif_sse_flank=motif_sse_flank,
        binder_sse_prop=binder_sse_prop,
        motif_placements_full=motif_placements_full,
        return_last=False,
        return_aux=True,
        dummy_fill_mode=model.config.data.dummy_fill_mode,
        xt_0_path=xt_0_path,
        xt_1_path=xt_1_path,
        **sampling_kwargs,
    )
    # account for possible override from partial diffusion input structure
    seq_mask = aux["seq_mask"]
    residue_index = aux["residue_index"]
    if sampling_kwargs["partial_diffusion"]["enabled"]:
        residue_index = residue_index_orig
    chain_index = aux["chain_index"]
    aux["chain_id_mapping"] = chain_id_mapping

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
            motif_sse_label=motif_sse_label,
            binder_sse_prop=binder_sse_prop,
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


def _should_keep_atom(resname: str, atom_name: str, strategy: str) -> bool:
    """Check if an atom should be kept for the given conditioning strategy."""
    tip_atoms = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS
    bb_atoms = {"N", "CA", "C", "O"}
    if "acidic-tip" in strategy:
        # Base: ASP/GLU tip atoms only.
        # Variant "+<aa3>-full" suffix(es): listed residue types get all atoms kept.
        if resname in ("ASP", "GLU") and atom_name in tip_atoms.get(resname, []):
            return True
        full_aa3_set = set()
        for token in strategy.split("+"):
            token = token.strip().lower()
            if token.endswith("-full"):
                full_aa3_set.add(token[: -len("-full")].upper())
        if resname in full_aa3_set:
            return True
        return False
    elif "sidechain-tip" in strategy:
        return atom_name in tip_atoms.get(resname, [])
    elif "backbone-sidechain" in strategy:
        return True
    elif "backbone" in strategy:
        return atom_name in bb_atoms
    elif "sidechain" in strategy:
        return atom_name not in bb_atoms
    return True


def _write_conditioning_atoms_pdb(
    motif_fp: Path, save_dir: Path, strategy: str
) -> None:
    """Write a filtered PDB containing only atoms used for conditioning."""
    out_path = save_dir / f"{Path(motif_fp).stem}_conditioning_atoms.pdb"
    with open(motif_fp) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if not line.startswith(("ATOM", "HETATM")):
                if line.startswith("END"):
                    f_out.write(line)
                continue
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            if _should_keep_atom(resname, atom_name, strategy):
                f_out.write(line)
        f_out.write("END\n")


def _transform_pdb_line(line: str, centering_delta: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> str:
    """Apply centering, rotation, and translation to ATOM/HETATM line coordinates."""
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    coords = np.array([x, y, z]) - centering_delta
    coords = rotation @ coords
    coords = coords + translation
    return f"{line[:30]}{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}{line[54:]}"


def _write_transformed_motif_pdbs(
    motif_fp: Path,
    save_dir: Path,
    sample_idx: int,
    centering_delta: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    strategy: str | None = None,
) -> None:
    """Write per-sample transformed motif PDBs (full + conditioning atoms only)."""
    stem = Path(motif_fp).stem
    full_path = save_dir / f"motif_rotated_{sample_idx:02d}.pdb"
    cond_path = save_dir / f"motif_rotated_{sample_idx:02d}_conditioning_atoms.pdb"
    with open(motif_fp) as f_in:
        lines = f_in.readlines()
    with open(full_path, "w") as f_full, open(cond_path, "w") as f_cond:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                transformed = _transform_pdb_line(line, centering_delta, rotation, translation)
                f_full.write(transformed)
                if strategy is not None:
                    resname = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    if _should_keep_atom(resname, atom_name, strategy):
                        f_cond.write(transformed)
            elif line.startswith("END"):
                f_full.write(line)
                f_cond.write(line)
        if not any(l.startswith("END") for l in lines):
            f_full.write("END\n")
            f_cond.write("END\n")


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
    motif_sse_label=None,
    motif_sse_flank=None,
    binder_sse_prop=None,
    run_name="",
    allatom=False,
    xt_0_path: Path | None = None,
    xt_1_path: Path | None = None
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
    all_chain_id_mappings = []
    seq_mask = []
    atom_mask = []
    motif_idx = []
    motif_coords = []
    motif_aatypes = []
    motif_atom_mask = []
    motif_random_rots = []
    motif_centering_deltas = []
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

    sample_offset = 0
    for i, bs in enumerate(batch_sizes):

        si, ei = sample_offset, sample_offset + bs

        if fixed_motif_pos is not None:
            sampling_config["sampling"]["conditional_cfg"][
                "discontiguous_motif_assignment"
            ]["fixed_motif_pos"] = fixed_motif_pos[si:ei]

        # Create batch-specific copies to avoid overwriting the original tensors
        sse_cond_batch = None
        adj_cond_batch = None
        if sse_cond is not None and adj_cond is not None:
            sse_cond_batch = torch.cat([sse_cond.clone() for _ in range(bs)], dim=0)
            adj_cond_batch = torch.cat([adj_cond.clone() for _ in range(bs)], dim=0)

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
                sse_cond=sse_cond_batch,
                adj_cond=adj_cond_batch,
                motif_sse_label=motif_sse_label,
                binder_sse_prop=binder_sse_prop,
                motif_placements_full=(
                    motif_placements_full[si:ei]
                    if motif_placements_full is not None
                    else None
                ),
                xt_0_path=xt_0_path,
                xt_1_path=xt_1_path,
                **curr_sampling_config,
            )
        )
        trimmed_coords.extend(trimmed_coords_bi)
        trimmed_residue_index.extend(trimmed_residue_index_bi)
        trimmed_chain_index.extend(trimmed_chain_index_bi)
        all_chain_id_mappings.extend(samp_aux_bi["chain_id_mapping"])
        seq_mask.extend(seq_mask_bi)
        if samp_aux_bi["motif_idx"] is not None:
            motif_idx.extend(samp_aux_bi["motif_idx"])
            motif_aatypes.extend(samp_aux_bi["motif_aatype"])
            motif_coords.append(samp_aux_bi["motif_all_atom"])
            motif_atom_mask.extend(samp_aux_bi["motif_atom_mask"])
        if samp_aux_bi.get("motif_random_rots") is not None:
            motif_random_rots.append(samp_aux_bi["motif_random_rots"])
        if samp_aux_bi.get("motif_centering_delta") is not None:
            motif_centering_deltas.append(samp_aux_bi["motif_centering_delta"])
        runtime += samp_aux_bi["runtime"]
        sequences.extend(samp_aux_bi["s"])
        atom_mask.extend(samp_aux_bi["atom_mask"])

        sample_offset += bs

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
        "motif_random_rots": torch.cat(motif_random_rots, dim=0) if motif_random_rots else None,
        "motif_centering_delta": motif_centering_deltas[0] if motif_centering_deltas else None,
        "s": sequences,
        "runtime": runtime,
        "all_chain_id_mappings": all_chain_id_mappings,
    }

    sampled_sequences = None
    if allatom:
        sampled_sequences = [
            "".join([
                residue_constants.restypes[aaint] if aaint < len(residue_constants.restypes) else "X"
                for aaint in s_hat_bi
            ])
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
    motif_dir: Path = Path("examples/motifs/nanobody"),
    motif_pdb: Path | None = None,
    motif_contig_override: str | None = None,
    num_samples: int = 8,
    num_mpnn_seqs: int = 8,
    batch_size: int = 64,
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
        motif_pdb (Path, optional): Overrides motif_dir, use by specifying the motifs inside sampling_yaml_path to null
        motif_contig_override (str, optional): If specified, overrides motif_contig in sampling_yaml_path for all motifs. Defaults to None.
        num_samples (int, optional): Total number of samples to draw. Defaults to 8.
        num_mpnn_seqs (int, optional): If 0, skips sequence design and ESMFold evaluation. Defaults to 8.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
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
    else:
        search_space["rewind_steps"] = [None]

    latent_interpolation = (
        "latent_interpolation" in runner_cfg and runner_cfg["latent_interpolation"]["enabled"]
    )
    if latent_interpolation:
        latent_dir = Path(runner_cfg["latent_interpolation"]["latent_dir"])
        search_space["latent_paths"] = [[latent_dir / latent_pt[0], latent_dir / latent_pt[1]] for latent_pt in runner_cfg["latent_interpolation"]["latent_paths"]]
    else:
        search_space["latent_paths"] = [[None, None]]

    # Post-hoc EMA: phema_stds is the last key in the unpacking tuple, so move it
    # to the end if a YAML inserted it earlier. Default [None] means no EMA reconstruction.
    phema_stds = search_space.pop("phema_stds", [None])
    search_space["phema_stds"] = phema_stds

    # Inference-time guidance (branch-evaluate-select). A single global block
    # in the runner YAML — applies to every search_space point and every motif.
    inference_guidance_cfg = runner_cfg.get("inference_guidance")

    if motif_contig_override is not None:
        length_range_override = []
        for chain_contig in motif_contig_override.split(";/;"):
            curr_chain_length = 0
            for segment in chain_contig.split(";"):
                if segment[0].isalpha():
                    seg_start, seg_end = tuple(segment[1:].split("-"))
                    curr_chain_length += (int(seg_end) - int(seg_start) + 1)
                else:
                    seg_start, seg_end = tuple(segment.split("-"))
                    assert seg_end == seg_start, "scaffold segments must not have flexible ranges when using motif contig override."
                    curr_chain_length += int(seg_end)
            length_range_override.append([curr_chain_length, curr_chain_length])
        print(f"Using motif contig override: {motif_contig_override} with length ranges {length_range_override}")

    motif_fps = []
    motif_contigs = []
    scaffold_lengths = []
    hotspots = []
    ssadj_fps = []
    motif_sse_labels = []
    binder_sse_props = []
    all_save_dirs = []

    motif_sse_field = runner_cfg.get("motif_sse")
    n_motifs = len(runner_cfg["motifs"])
    if motif_sse_field is None:
        motif_sse_field = [None] * n_motifs
    elif len(motif_sse_field) != n_motifs:
        raise ValueError(
            f"motif_sse list has length {len(motif_sse_field)} but expected {n_motifs}"
        )

    binder_sse_prop_field = runner_cfg.get("binder_sse_prop")
    if binder_sse_prop_field is None:
        binder_sse_prop_field = [None] * n_motifs
    elif len(binder_sse_prop_field) != n_motifs:
        raise ValueError(
            f"binder_sse_prop list has length {len(binder_sse_prop_field)} but expected {n_motifs}"
        )

    # Optional per-motif SSE flanking (cc146/cc147). When set to an int N > 0, the
    # model.sample() builder extends motif SSE one-hot to the N residues on either
    # side of each motif residue (within the same chain, within seq_mask), so the
    # caller can dictate motif-adjacent secondary structure (e.g. helix flanks a
    # helical motif). A single int broadcasts to all motifs.
    motif_sse_flank_field = runner_cfg.get("motif_sse_flank")
    if motif_sse_flank_field is None:
        motif_sse_flank_field = [None] * n_motifs
    elif isinstance(motif_sse_flank_field, int):
        motif_sse_flank_field = [motif_sse_flank_field] * n_motifs
    elif len(motif_sse_flank_field) != n_motifs:
        raise ValueError(
            f"motif_sse_flank list has length {len(motif_sse_flank_field)} but expected {n_motifs}"
        )

    motif_sse_flanks = []
    for ri, (motif_cfg, motif_contig, length_range, hs, ssadj, sse_lbl, bprop, sse_flank) in enumerate(
        zip(
            runner_cfg["motifs"],
            runner_cfg["motif_contigs"],
            runner_cfg["total_lengths"],
            runner_cfg["hotspots"],
            runner_cfg["ssadj"],
            motif_sse_field,
            binder_sse_prop_field,
            motif_sse_flank_field,
        )
    ):
        if motif_cfg is not None and motif_pdb is not None:
            raise ValueError("Only one of motif_cfg or motif_pdb can be specified.")
        if motif_cfg is None:
            if motif_pdb is not None:
                motif_fps.append(motif_pdb)
            else:
                motif_fps.append(motif_dir / f"{ri:03}_unconditional")
        else:
            if isinstance(motif_cfg, list):
                # Check if it's the new named format: ["name", [["motif1", w1], ...]]
                # vs old format: [["motif1", w1], ["motif2", w2], ...]
                if (len(motif_cfg) == 2
                    and isinstance(motif_cfg[0], str)
                    and isinstance(motif_cfg[1], list)
                    and len(motif_cfg[1]) > 0
                    and isinstance(motif_cfg[1][0], list)):
                    # New named format: ["name", [["motif1", w1], ...]]
                    multi_motif_name = motif_cfg[0]
                    motif_list = motif_cfg[1]
                else:
                    # Old format: [["motif1", w1], ["motif2", w2], ...]
                    multi_motif_name = None
                    motif_list = motif_cfg

                multi_motif_entries = []
                for entry in motif_list:
                    name, weight = entry[0], entry[1]
                    if not name.endswith('.pdb') and not name.endswith('.cif'):
                        path = motif_dir / f"{name}.pdb"
                    else:
                        path = motif_dir / name
                    multi_motif_entries.append((path, weight))
                # Store as tuple: (name_or_None, list of (path, weight))
                motif_fps.append((multi_motif_name, multi_motif_entries))
            else:
                # Single motif (existing logic)
                if not motif_cfg.endswith('.pdb') and not motif_cfg.endswith('.cif'):
                    motif_fps.append(motif_dir / f"{motif_cfg}.pdb")
                else:
                    motif_fps.append(motif_dir / motif_cfg)
        if motif_contig_override is not None:
            motif_contigs.append(motif_contig_override)
            scaffold_lengths.append(length_range_override)
        else:
            motif_contigs.append(motif_contig)
            scaffold_lengths.append(length_range)
        hotspots.append(hs)
        ssadj_fps.append(ssadj)
        motif_sse_labels.append(sse_lbl)
        binder_sse_props.append(bprop)
        motif_sse_flanks.append(sse_flank)

    # get all params, and optionally split into chunks for parallelization
    all_params = list(itertools.product(*search_space.values()))
    if array_id is not None:
        chunk_size = math.ceil(len(all_params) / num_arrays)
        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_params))
        all_params = all_params[start_idx:end_idx]

    # for a given setting
    for curr_params in tqdm(all_params, "Evaluating search space"):

        ((model_name, epoch, sampling_config_name),
            stepscale,
            schurn,
            cc_start,
            (dx, dy, dz),
            rs,
            (xt_0_path, xt_1_path),
            phema_std,
        ) = curr_params

        dxs = f"{dx:.1f}"
        dys = f"{dy:.1f}"
        dzs = f"{dz:.1f}"

        phema_suffix = f"-phema{phema_std:.3f}" if phema_std is not None else ""
        if latent_interpolation:
            save_suffix = f"{model_name}-epoch{epoch}-{sampling_config_name}-ss{stepscale}-schurn{schurn}-ccstart{cc_start}-dx{dxs}-dy{dys}-dz{dzs}-rewind{rs}{phema_suffix}-{xt_0_path.stem}-{xt_1_path.stem}"
        else:
            save_suffix = f"{model_name}-epoch{epoch}-{sampling_config_name}-ss{stepscale}-schurn{schurn}-ccstart{cc_start}-dx{dxs}-dy{dys}-dz{dzs}-rewind{rs}{phema_suffix}"

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
            motif_sse_label,
            binder_sse_prop,
            motif_sse_flank,
        ) in zip(
            motif_fps,
            motif_contigs,
            scaffold_lengths,
            hotspots,
            ssadj_fps,
            motif_sse_labels,
            binder_sse_props,
            motif_sse_flanks,
        ):
            if isinstance(motif_fp, tuple) and len(motif_fp) == 2 and isinstance(motif_fp[1], list):
                # Multi-motif format: (name_or_None, [(path, weight), ...])
                multi_motif_name, multi_motif_entries = motif_fp
                if multi_motif_name is not None:
                    # Use provided name
                    combined_name = multi_motif_name
                else:
                    # Generate name from motif names
                    motif_names = [Path(p).stem for p, _ in multi_motif_entries]
                    combined_name = "__".join(motif_names[:3])
                    if len(motif_names) > 3:
                        combined_name += f"__and_{len(motif_names) - 3}_more"
                per_motif_save_dir = per_config_save_dir / combined_name
                if save_shortname:
                    save_name = combined_name
                # Update motif_fp to just the entries for downstream processing
                motif_fp = multi_motif_entries
            else:
                per_motif_save_dir = (
                    per_config_save_dir / motif_fp.stem
                )  # one config, one motif
                if save_shortname:
                    save_name = f"{motif_fp.stem}"
            # Disambiguate save dir when the same motif name is used multiple times
            # with different SSE conditioning (e.g. helix vs strand variants).
            if motif_sse_label is not None:
                sse_tag = str(motif_sse_label).strip().replace("/", "_")
                per_motif_save_dir = per_motif_save_dir.with_name(
                    f"{per_motif_save_dir.name}__sse-{sse_tag}"
                )
                if save_shortname:
                    save_name = f"{save_name}__sse-{sse_tag}"
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
            if isinstance(motif_fp, list):
                # Multi-motif: pass list of (path, weight) tuples
                sampling_config["sampling"]["motif_file_path"] = None
                sampling_config["sampling"]["multi_motif_list"] = [
                    (str(path), weight) for path, weight in motif_fp
                ]
                # Copy all motif files for reference
                for path, _ in motif_fp:
                    if Path(path).exists():
                        shutil.copy(path, per_motif_save_dir)
            elif partial_diffusion:
                shutil.copy(motif_fp, per_motif_save_dir)
                sampling_config["sampling"]["partial_diffusion"][
                    "pdb_file_path"
                ] = motif_fp
                sampling_config["sampling"]["partial_diffusion"]["n_steps"] = rs
                if motif_contig is None:
                    sampling_config["sampling"]["motif_file_path"] = "test_dir/empty.pdb"
                else:
                    sampling_config["sampling"]["motif_file_path"] = motif_fp
                sampling_config["sampling"]["multi_motif_list"] = None
            else:
                sampling_config["sampling"]["motif_file_path"] = motif_fp
                sampling_config["sampling"]["multi_motif_list"] = None
                # Unconditional runs (motifs: [null]) use a placeholder path
                # (motif_dir/"NNN_unconditional") that never points at a real
                # file, so skip the provenance copy / conditioning-atoms write
                # rather than crashing on a missing file.
                if Path(motif_fp).exists():
                    # Copy original motif PDB and filtered conditioning-atoms PDB
                    shutil.copy(motif_fp, per_motif_save_dir)
                    cond_strategy = sampling_config["sampling"]["conditional_cfg"][
                        "crop_conditional_guidance"
                    ].get("strategy")
                    if cond_strategy is not None:
                        _write_conditioning_atoms_pdb(
                            motif_fp, per_motif_save_dir, cond_strategy
                        )
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
            # Unindexed conditioning: motif is not placed at scaffold indices, so we
            # skip contig parsing entirely. The sample() path picks the total length
            # from length_ranges_per_chain and the model appends motif tokens along
            # the sequence dim internally.
            is_unindexed_contig = motif_contig == "unindexed"
            if is_unindexed_contig:
                motif_placements = ["unindexed" for _ in range(num_samples)]
            elif (
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

            cache_key = (config_path, checkpoint_path, phema_std)
            if model_info == (None, None) or model_info != cache_key:
                if phema_std is not None:
                    from protpardelle.core.models import load_model_phema
                    # Look for model-specific EMA snapshot dir first (avoids
                    # collisions when many model checkpoints share oak/weights/),
                    # falling back to the flat weights dir for backward compat.
                    weights_dir = Path(checkpoint_path).parent
                    candidate_dirs = [
                        weights_dir / f"{model_name}_phema",
                        weights_dir,
                    ]
                    checkpoint_dir = next(
                        (d for d in candidate_dirs if d.is_dir()), weights_dir
                    )
                    model = load_model_phema(
                        config_path, checkpoint_dir, int(epoch), phema_std
                    )
                else:
                    model = load_model(config_path, checkpoint_path)
                model_info = cache_key

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
                motif_sse_label=motif_sse_label,
                motif_sse_flank=motif_sse_flank,
                binder_sse_prop=binder_sse_prop,
                run_name=run_name,
                allatom=allatom,
                xt_0_path=xt_0_path,
                xt_1_path=xt_1_path,
            )

            curr_runtime = aux[-2]["runtime"]
            total_sampling_time += curr_runtime

            motif_name_for_log = combined_name if isinstance(motif_fp, list) else motif_fp.stem
            logger.info("Sampling %s took %.2f seconds.", motif_name_for_log, curr_runtime)

            # NB: scaffold_info.csv build/write moved to AFTER the
            # guidance + caliby_redenoise stage so the recorded
            # matched_motif_idx / matched_motif_aa3 reflect the WINNER branch
            # (with its own linear_sum_assignment), not the main sample's
            # pre-guidance matching. See block after caliby_redenoise.
            df_scaffold_info = defaultdict(list)
            df_scaffold_info["sample_num"] = list(range(num_samples))
            if motif_contig is not None:
                df_scaffold_info["motif_placements"] = motif_placements

            curr_samp_save_names, curr_pred_save_names = save_samples(
                aux,
                per_motif_save_dir,
                save_name,
                bb_only=model.config.model.task == "backbone",
                allatom=allatom,
                motif_placements=motif_placements_full,
            )

            # Save per-sample motif_rotated PDBs immediately after main samples are
            # written, so the motif context is on disk even if a later step crashes.
            _samp_aux_data_early = aux[-2]
            if (
                not isinstance(motif_fp, list)
                and _samp_aux_data_early.get("motif_random_rots") is not None
                and _samp_aux_data_early.get("motif_centering_delta") is not None
            ):
                _rots = _samp_aux_data_early["motif_random_rots"].cpu().numpy()
                _cd = _samp_aux_data_early["motif_centering_delta"][0, 0, 0].cpu().numpy()
                _dx = sampling_config["sampling"].get("dx", 0.0) or 0.0
                _dy = sampling_config["sampling"].get("dy", 0.0) or 0.0
                _dz = sampling_config["sampling"].get("dz", 0.0) or 0.0
                _trans = np.array([float(_dx), float(_dy), float(_dz)])
                _cond_strategy = sampling_config["sampling"]["conditional_cfg"][
                    "crop_conditional_guidance"
                ].get("strategy")
                for _si in range(num_samples):
                    _write_transformed_motif_pdbs(
                        motif_fp, per_motif_save_dir, _si,
                        centering_delta=_cd,
                        rotation=_rots[_si],
                        translation=_trans,
                        strategy=_cond_strategy,
                    )
            _motif_rotated_already_written = True

            # Caliby ensemble sequence design — replaces MPNN-assigned sequences
            # with a single Caliby-designed sequence per main sample, conditioned
            # on an ensemble generated via partial_diffusion from each main PDB.
            # After Caliby returns a sequence, re-denoise the main sample from step
            # begin_mpnn_step -> n_steps with gt_aatype fixed to Caliby's sequence,
            # so the final all-atom coords are consistent with Caliby's seq.
            caliby_cfg = sampling_config["sampling"]["allatom_cfg"].get("caliby_ensemble", {})

            # Inference-time guidance: branch-evaluate-select. Each primary is
            # forked into n_branches via partial_diffusion rewind, single-backbone
            # Caliby scores each, top-1 winner overwrites the primary PDB + aux.
            # When enabled, the winner's single-backbone Caliby sequence is fed
            # directly into caliby_redenoise (replacing the bias-unaware 32-decoy
            # ensemble Caliby step), so aatype_biases survive to the final PDB.
            guidance_enabled = (
                allatom
                and inference_guidance_cfg is not None
                and inference_guidance_cfg.get("enabled", False)
            )
            guidance_winner_seqs: list[str] | None = None
            if guidance_enabled:
                guidance_tmp_root = per_motif_save_dir / "inference_guidance"
                guidance_tmp_root.mkdir(exist_ok=True)
                guidance_winner_seqs = run_inference_guidance(
                    model=model,
                    main_sample_pdbs=curr_samp_save_names,
                    aux=aux,
                    sampling_config=sampling_config,
                    guidance_cfg=inference_guidance_cfg,
                    caliby_cfg=caliby_cfg,
                    tmp_root=str(guidance_tmp_root),
                    motif_placements_full=(
                        None if is_unindexed_contig else motif_placements_full
                    ),
                    is_unindexed_contig=is_unindexed_contig,
                    # Args for rewriting motif_rotated_NN.pdb with winner's rotation
                    motif_fp=(
                        motif_fp if not isinstance(motif_fp, list) else None
                    ),
                    motif_translation=(float(dx), float(dy), float(dz)),
                    per_motif_save_dir=per_motif_save_dir,
                )

            if allatom and caliby_cfg.get("enabled", False):
                if guidance_enabled and guidance_winner_seqs is not None:
                    # Skip the 32-decoy ensemble Caliby; use the guidance
                    # winners' single-backbone sequences directly. They already
                    # have aatype_biases-selected prolines AND the motif AAs
                    # pinned via fixed_pos_override_seq.
                    caliby_seqs = guidance_winner_seqs
                    logger.info(
                        "[Caliby] guidance active: skipping ensemble pass, "
                        "using winner seqs for redenoise (%d primaries)",
                        len(caliby_seqs),
                    )
                else:
                    caliby_tmp_root = per_motif_save_dir / "caliby_ensembles"
                    caliby_tmp_root.mkdir(exist_ok=True)
                    caliby_seqs = run_caliby_ensemble_design(
                        model=model,
                        main_sample_pdbs=curr_samp_save_names,
                        aux=aux,
                        motif_fp=motif_fp,
                        motif_placements_full=motif_placements_full,
                        sampling_config=sampling_config,
                        caliby_cfg=caliby_cfg,
                        tmp_root=str(caliby_tmp_root),
                        is_unindexed_contig=is_unindexed_contig,
                    )
                # Third pass: partial_diffusion with gt_aatype = Caliby seq to
                # properly denoise all atom types for the Caliby sequence.
                caliby_redenoise(
                    model=model,
                    main_sample_pdbs=curr_samp_save_names,
                    caliby_seqs=caliby_seqs,
                    aux=aux,
                    sampling_config=sampling_config,
                    caliby_cfg=caliby_cfg,
                    motif_placements_full=(
                        None if is_unindexed_contig else motif_placements_full
                    ),
                )
                logger.info("[Caliby] re-denoised %d main PDBs with Caliby sequences", len(caliby_seqs))

            # Write scaffold_info AFTER guidance + caliby_redenoise so the
            # matched_motif_idx / matched_motif_aa3 columns reflect the WINNER's
            # linear_sum_assignment (and therefore align with what's actually
            # in the on-disk sample_NN.pdb).
            if is_unindexed_contig:
                samp_aux_for_idx = aux[-2]
                trimmed_residue_index_for_idx = aux[1]
                matched_idx = samp_aux_for_idx.get("motif_idx")
                matched_aatypes = samp_aux_for_idx.get("motif_aatype")
                if matched_idx is not None:
                    pdb_resnums = []
                    for i in range(num_samples):
                        ridx = trimmed_residue_index_for_idx[i]
                        pdb_resnums.append(",".join(
                            str(int(ridx[int(x)].item())) for x in matched_idx[i]
                        ))
                    df_scaffold_info["matched_motif_idx"] = pdb_resnums
                if matched_aatypes is not None:
                    from protpardelle.common import residue_constants as _rc
                    df_scaffold_info["matched_motif_aa3"] = [
                        ",".join(
                            _rc.restype_1to3.get(
                                _rc.restypes[int(a)], "UNK"
                            ) if int(a) < len(_rc.restypes) else "UNK"
                            for a in matched_aatypes[i]
                        )
                        for i in range(num_samples)
                    ]
            df_scaffold_info = pd.DataFrame(df_scaffold_info)
            df_scaffold_info.to_csv(
                per_motif_save_dir / "scaffold_info.csv", index=False
            )

            # (motif_rotated PDBs already written right after save_samples above)
            samp_aux_data = aux[-2]

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
    motif_pdb: Path | None = typer.Option(
        None, help="Path to a single motif PDB, overrides motif_dir"
    ),
    num_samples: int = typer.Option(8, help="Number of samples to draw"),
    num_mpnn_seqs: int = typer.Option(
        8, help="Number of sequences to design with MPNN (0 to skip)"
    ),
    batch_size: int = typer.Option(64, help="Batch size for sampling"),
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
        motif_pdb=motif_pdb,
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
