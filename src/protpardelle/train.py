"""Entrypoint for Protpardelle-1c training.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li
"""

import os
import shlex
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import typer
import wandb
import yaml
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import protpardelle.core.diffusion as diffusion
import protpardelle.core.models as models
import protpardelle.core.modules as modules
from protpardelle.common import residue_constants
from protpardelle.core.phema import PowerFunctionEMA
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.dataset import (
    PDBDataset,
    StochasticMixedSampler,
    calc_sigma_data,
    make_crop_cond_mask_and_recenter_coords,
)
from protpardelle.utils import (
    dict_to_namespace,
    expand_env_in_config,
    seed_everything,
    unsqueeze_trailing_dims,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def masked_cross_entropy(
    logprobs: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Compute the masked cross-entropy loss.

    Args:
        logprobs (torch.Tensor): Log probabilities of the predicted tokens.
        target (torch.Tensor): One-hot encoded target tokens.
        loss_mask (torch.Tensor): Mask to apply to the loss.

    Returns:
        torch.Tensor: The computed masked cross-entropy loss.
    """

    cel = -target * logprobs
    cel = cel * loss_mask[..., None]
    cel = cel.sum((-1, -2)) / loss_mask.sum(-1).clamp(min=1e-6)

    return cel


def masked_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the masked mean squared error loss.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values.
        mask (torch.Tensor): Mask to apply to the loss.
        weights (torch.Tensor | None, optional): Weights to apply to the loss. Defaults to None.
        tol (float, optional): Tolerance for the loss computation. Defaults to 1e-7.

    Returns:
        torch.Tensor: The computed masked mean squared error loss.
    """

    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * mask
    if weights is not None:
        mse = mse * unsqueeze_trailing_dims(weights, mse)
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=tol)

    return mse


class ProtpardelleRunner:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        save_dir,
        device,
        scaler=None,
        teacher_model=None,
    ):
        self.config = config

        if isinstance(model, (nn.DataParallel, DDP)):
            self.model = model.module
        else:
            self.model = model
        self.forward = model

        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()
        self.dataset = train_dataset
        self.save_dir = save_dir
        self.device = device
        self.scaler = scaler

        self.next_eval_time = config.train.eval_freq

        self.mpnn_model = None
        self.struct_pred_model = None
        self.tokenizer = None

        self.sigma_data = self.model.sigma_data

        self.teacher_model = teacher_model
        distill_cfg = getattr(config.train, "distill", None)
        self.distill_enabled = (
            teacher_model is not None
            and distill_cfg is not None
            and getattr(distill_cfg, "enabled", False)
        )
        self.distill_cfg = distill_cfg

    def get_optimizer_and_scheduler(self):
        params_to_train = [(n, p) for n, p in self.model.named_parameters()]
        if self.model.task == "seqdes":
            params_to_train = [
                (n, p) for n, p in params_to_train if "struct_model" not in n
            ]
        params_to_train = [p for n, p in params_to_train]
        optimizer = torch.optim.Adam(
            params_to_train,
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = modules.LinearWarmupCosineDecay(
            optimizer,
            self.config.train.lr,
            warmup_steps=self.config.train.warmup_steps,
            decay_steps=self.config.train.decay_steps,
        )
        return optimizer, scheduler

    def train_init(self):
        print(f"total params: {sum(p.numel() for p in self.model.parameters())}")
        print(
            f"trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def compute_distill_loss(
        self,
        inputs,
        return_aux=False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        """FastGen-style consistency distillation loss.

        Sample (sigma_t, sigma_r) with sigma_r < sigma_t. Use the frozen teacher to
        take an EDM Euler step from y_t (noised at sigma_t) to y_r (at sigma_r), then
        enforce student(y_t, sigma_t) ~= student(y_r, sigma_r).

        Reference: FastGen `CMModel._compute_cm_loss` with `loss_config.use_cd=True`
        (https://github.com/NVlabs/FastGen).
        """
        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        aatype = inputs["aatype"]
        atom_mask = inputs["atom_mask"]
        device = coords.device
        bs = coords.shape[0]

        is_unindexed = (
            self.config.train.crop_conditional
            and "unindexed" in self.config.model.conditioning_style
        )
        assert not is_unindexed, "Distillation does not support unindexed conditioning."

        struct_crop_cond = None
        crop_cond_mask = None
        hotspot_mask = None
        if self.config.train.crop_conditional:
            coords, crop_cond_mask, hotspot_mask = (
                make_crop_cond_mask_and_recenter_coords(
                    atom_mask,
                    coords,
                    aatype=aatype,
                    chain_index=inputs["chain_index"],
                    forced_motif_chain=inputs.get("forced_motif_chain"),
                    **vars(self.config.train.crop_cond),
                )
            )
            struct_crop_cond = coords * crop_cond_mask[..., None]

        sse_cond, adj_cond = None, None
        if "ssadj" in self.config.model.conditioning_style:
            sse_cond = inputs["sse"]
            adj_cond = inputs["adj"]

        # Sample (t, r) timesteps in (0, 1) with the FastGen sigmoid curriculum
        eps_t = 1e-9
        sigmoid_ratio = float(getattr(self.distill_cfg, "sigmoid_ratio", 0.0))
        weighting = getattr(self.distill_cfg, "weighting", "default")
        huber_const = float(getattr(self.distill_cfg, "huber_const", 0.0))
        min_r = float(getattr(self.distill_cfg, "min_r", 1e-6))
        t_step = torch.rand(bs, device=device).clamp(min=eps_t, max=1 - eps_t)
        r_step = t_step - t_step * (1 - sigmoid_ratio) * (
            1 + 8 * torch.sigmoid(-t_step)
        )
        r_step = torch.clamp(r_step, min=min_r)

        sigma_t = self.model.training_noise_schedule(t_step)
        sigma_r = self.model.training_noise_schedule(r_step)
        # Lognormal schedule is monotone in t, but guard against ties.
        sigma_t = torch.maximum(sigma_t, sigma_r * 1.001 + 1e-6)

        # Build noised y_t with shared epsilon (same dummy fill as noise_coords)
        dummy_fill_mask = 1.0 - atom_mask
        if self.config.data.dummy_fill_mode == "zero":
            dummy_fill_value = torch.zeros_like(coords)
        else:
            dummy_fill_value = coords[..., 1:2, :]  # CA
        coords_filled = (
            coords * atom_mask.unsqueeze(-1)
            + dummy_fill_value * dummy_fill_mask.unsqueeze(-1)
        )
        epsilon = torch.randn_like(coords_filled)
        sigma_t_b = unsqueeze_trailing_dims(sigma_t, coords_filled)
        sigma_r_b = unsqueeze_trailing_dims(sigma_r, coords_filled)
        y_t = coords_filled + epsilon * sigma_t_b

        forward_kwargs = dict(
            seq_mask=seq_mask,
            residue_index=inputs["residue_index"],
            chain_index=inputs["chain_index"],
            hotspot_mask=(
                hotspot_mask
                if "hotspot" in self.config.model.conditioning_style
                else None
            ),
            struct_crop_cond=struct_crop_cond,
            crop_cond_mask=(
                crop_cond_mask
                if self.config.train.crop_conditional
                and "condmask" in self.config.model.conditioning_style
                else None
            ),
            sse_cond=sse_cond,
            adj_cond=adj_cond,
        )

        # Teacher Euler step: y_r = y_t + (sigma_r - sigma_t) * (y_t - D_T) / sigma_t
        with torch.no_grad():
            D_T_yt, _, _, _ = self.teacher_model(
                noisy_coords=y_t, noise_level=sigma_t, **forward_kwargs
            )
            flow = (y_t - D_T_yt) / sigma_t_b.clamp(min=1e-9)
            y_r = y_t + (sigma_r_b - sigma_t_b) * flow

        # Student forward (with grad) at (y_t, sigma_t)
        D_S_yt, _, _, _ = self.forward(
            noisy_coords=y_t, noise_level=sigma_t, **forward_kwargs
        )
        # Student forward (frozen target) at (y_r, sigma_r)
        with torch.no_grad():
            D_S_yr, _, _, _ = self.forward(
                noisy_coords=y_r, noise_level=sigma_r, **forward_kwargs
            )

        struct_loss_mask = torch.ones_like(coords_filled) * atom_mask[..., None]
        diff_sq = (D_S_yt - D_S_yr).pow(2) * struct_loss_mask
        data_dims = tuple(range(1, len(diff_sq.shape)))
        l2_sq = diff_sq.sum(data_dims) / struct_loss_mask.sum(data_dims).clamp(min=1e-7)
        if huber_const > 0:
            loss_unweighted = torch.sqrt(l2_sq + huber_const**2) - huber_const
        else:
            loss_unweighted = l2_sq

        if weighting == "default":
            weight = 1.0 / (sigma_t - sigma_r).clamp(min=1e-6)
        elif weighting == "one":
            weight = torch.ones_like(sigma_t)
        elif weighting == "sigma_sq":
            weight = 1.0 / sigma_t.pow(2).clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown distill weighting: {weighting}")

        cd_loss = (loss_unweighted * weight).mean()

        aux = {
            "distill_loss": cd_loss.detach().cpu().item(),
            "distill_loss_unweighted": loss_unweighted.mean().detach().cpu().item(),
            "sigma_t_mean": sigma_t.mean().detach().cpu().item(),
            "sigma_r_mean": sigma_r.mean().detach().cpu().item(),
            "train_loss": cd_loss.detach().cpu().item(),
        }
        if return_aux:
            return cd_loss, aux
        return cd_loss

    def compute_loss(
        self,
        inputs,
        timestep=None,
        is_training=False,
        return_aux=False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        if self.distill_enabled and is_training:
            return self.compute_distill_loss(inputs, return_aux=return_aux)
        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        aatype = inputs["aatype"]
        aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
        atom_mask = inputs["atom_mask"]
        device = coords.device
        bs = coords.shape[0]

        # Initialize variables that may not be set in all branches
        struct_crop_cond = None
        is_unindexed = (
            self.config.train.crop_conditional
            and "unindexed" in self.config.model.conditioning_style
        )
        if self.config.train.crop_conditional:
            assert (
                not self.config.model.compute_loss_on_all_atoms
            ), "Crop conditioning with compute_loss_on_all_atoms not implemented"
            coords, crop_cond_mask, hotspot_mask = (
                make_crop_cond_mask_and_recenter_coords(
                    atom_mask,
                    coords,
                    aatype=aatype,
                    chain_index=inputs["chain_index"],
                    forced_motif_chain=inputs.get("forced_motif_chain"),
                    **vars(self.config.train.crop_cond),
                )
            )
            if not is_unindexed:
                struct_crop_cond = coords * crop_cond_mask[..., None]

        # For unindexed conditioning: extract motif residues and append to sequence
        # Save protein-only tensors for loss computation
        protein_N = coords.shape[1]
        motif_clean_coords = None
        if is_unindexed:
            M = self.config.model.max_motif_size
            motif_res_mask = crop_cond_mask.any(-1) > 0  # [B, N]

            # Extract motif residues, pad to M
            motif_clean_coords = torch.zeros(
                bs, M, *coords.shape[2:], device=device, dtype=coords.dtype
            )
            motif_condmask = torch.zeros(
                bs, M, crop_cond_mask.shape[-1], device=device, dtype=crop_cond_mask.dtype
            )
            motif_seq_mask = torch.zeros(bs, M, device=device, dtype=seq_mask.dtype)

            for bi in range(bs):
                motif_idx = torch.nonzero(motif_res_mask[bi]).squeeze(-1)
                n_m = min(len(motif_idx), M)
                if n_m > 0:
                    # Clean coords masked to conditioned atoms only
                    motif_clean_coords[bi, :n_m] = (
                        coords[bi, motif_idx[:n_m]]
                        * crop_cond_mask[bi, motif_idx[:n_m], :, None]
                    )
                    motif_condmask[bi, :n_m] = crop_cond_mask[bi, motif_idx[:n_m]]
                    motif_seq_mask[bi, :n_m] = 1.0

            # Save protein-only references for loss
            protein_coords = coords
            protein_seq_mask = seq_mask
            protein_atom_mask = atom_mask

            # Build extended tensors for model input
            seq_mask = torch.cat([seq_mask, motif_seq_mask], dim=1)
            atom_mask = torch.cat([atom_mask, motif_condmask], dim=1)
            crop_cond_mask = torch.cat([
                torch.zeros_like(crop_cond_mask),  # protein: no cond signal
                motif_condmask,
            ], dim=1)

        sse_cond, adj_cond = None, None
        if (
            "conditioning_style" in self.config.model
            and "ssadj" in self.config.model.conditioning_style
        ):
            sse_cond = inputs["sse"]
            adj_cond = inputs["adj"]

        # Per-residue DSSP one-hot conditioning at motif positions (cc143/cc144).
        # 50% of the time per sample, the SSE labels are dropped (zeroed out).
        # Shared keep_b is reused by binder_sse_prop_cond below so motif and binder
        # SSE channels are dropped together (CFG-coherent).
        keep_b = None
        motif_sse_cond = None
        if (
            self.config.train.crop_conditional
            and ("dssp" in self.config.model.conditioning_style
                 or "binderprop" in self.config.model.conditioning_style
                 or "sasa" in self.config.model.conditioning_style)
            and ("sse_dssp" in inputs or "sasa_class" in inputs)
        ):
            keep = (torch.rand(bs, device=device) >= 0.5).float()  # [B] — 50% kept
            keep_b = keep.view(bs, 1, 1)

        # Optional flanking-residue SSE conditioning (cc146/cc147). For each sample,
        # sample fl, fr ~ U{1, motif_dssp_flank_max} and mark the residues within
        # fl/fr of any motif residue (within the same chain, on the scaffold side)
        # as flanking. Those positions receive their DSSP one-hot in addition to
        # the motif residues, letting sampling specify SSE context beyond the motif.
        flank_max = int(getattr(self.config.model, "motif_dssp_flank_max", 0))
        flank_only_mask = None
        if (
            flank_max > 0
            and self.config.train.crop_conditional
            and "dssp" in self.config.model.conditioning_style
            and "sse_dssp" in inputs
        ):
            if is_unindexed:
                motif_in_protein_np = motif_res_mask.detach().cpu().numpy().astype(bool)
            else:
                motif_in_protein_np = (crop_cond_mask.any(-1) > 0).detach().cpu().numpy()
            chain_np = inputs["chain_index"].detach().cpu().numpy()
            seq_mask_np = inputs["seq_mask"].detach().cpu().numpy()
            flank_only_np = np.zeros((bs, protein_N), dtype=np.float32)
            for bi in range(bs):
                fl = int(np.random.randint(1, flank_max + 1))
                fr = int(np.random.randint(1, flank_max + 1))
                motif_pos = np.where(motif_in_protein_np[bi])[0]
                for mi in motif_pos:
                    chain_mi = chain_np[bi, mi]
                    for j in range(1, fl + 1):
                        mj = mi - j
                        if mj < 0:
                            break
                        if chain_np[bi, mj] != chain_mi:
                            break
                        if seq_mask_np[bi, mj] < 0.5:
                            break
                        if not motif_in_protein_np[bi, mj]:
                            flank_only_np[bi, mj] = 1.0
                    for j in range(1, fr + 1):
                        mj = mi + j
                        if mj >= protein_N:
                            break
                        if chain_np[bi, mj] != chain_mi:
                            break
                        if seq_mask_np[bi, mj] < 0.5:
                            break
                        if not motif_in_protein_np[bi, mj]:
                            flank_only_np[bi, mj] = 1.0
            flank_only_mask = torch.from_numpy(flank_only_np).to(device)

        if (
            self.config.train.crop_conditional
            and "dssp" in self.config.model.conditioning_style
            and "sse_dssp" in inputs
        ):
            sse_dssp_oh = inputs["sse_dssp"].float()  # [B, N, 3]
            if is_unindexed:
                # Extract motif SSE labels at motif residue positions, pad to M,
                # and append along seq dim so it lines up with the appended motif coords.
                motif_sse_at_motif = torch.zeros(
                    bs, M, 3, device=device, dtype=sse_dssp_oh.dtype
                )
                for bi in range(bs):
                    motif_idx = torch.nonzero(motif_res_mask[bi]).squeeze(-1)
                    n_m = min(len(motif_idx), M)
                    if n_m > 0:
                        motif_sse_at_motif[bi, :n_m] = sse_dssp_oh[bi, motif_idx[:n_m]]
                motif_sse_at_motif = motif_sse_at_motif * keep_b
                # Flank SSE at protein-body positions (zero where no flanking).
                if flank_only_mask is not None:
                    flank_sse_at_protein = sse_dssp_oh * flank_only_mask.unsqueeze(-1)
                    flank_sse_at_protein = flank_sse_at_protein * keep_b
                else:
                    flank_sse_at_protein = torch.zeros(
                        bs, protein_N, 3, device=device, dtype=sse_dssp_oh.dtype
                    )
                motif_sse_cond = torch.cat(
                    [flank_sse_at_protein, motif_sse_at_motif],
                    dim=1,
                )
            else:
                # Indexed: keep SSE at motif positions plus optional flank positions.
                motif_res_mask_indexed = (
                    crop_cond_mask.any(-1).float() if crop_cond_mask is not None else None
                )
                if motif_res_mask_indexed is not None:
                    combined_mask = motif_res_mask_indexed
                    if flank_only_mask is not None:
                        combined_mask = (combined_mask + flank_only_mask).clamp(max=1.0)
                    motif_sse_cond = sse_dssp_oh * combined_mask.unsqueeze(-1)
                else:
                    motif_sse_cond = torch.zeros_like(sse_dssp_oh)
                motif_sse_cond = motif_sse_cond * keep_b

        # Binder global SSE-proportion conditioning (cc145). Compute the (H, E, L)
        # fraction over non-motif residues from the same sse_dssp table, quantize to
        # 0.1 bins, then broadcast across binder positions. CFG dropout shares the
        # same `keep_b` draw as motif_sse_cond so when SSE is dropped, both channels
        # see zeros consistently.
        binder_sse_prop_cond = None
        if (
            self.config.train.crop_conditional
            and "binderprop" in self.config.model.conditioning_style
            and "sse_dssp" in inputs
        ):
            sse_dssp_oh_bp = inputs["sse_dssp"].float()  # [B, N, 3]
            seq_mask_pb = inputs["seq_mask"].float()  # protein-only [B, N]
            if is_unindexed:
                motif_res_mask_pb = motif_res_mask.float()  # [B, N]
            else:
                motif_res_mask_pb = (
                    crop_cond_mask.any(-1).float() if crop_cond_mask is not None else None
                )
            if motif_res_mask_pb is None:
                binder_res_mask = seq_mask_pb
            else:
                binder_res_mask = seq_mask_pb * (1.0 - motif_res_mask_pb)  # [B, N]
            # Sum H/E/L over binder positions, normalize by binder length.
            binder_counts = (sse_dssp_oh_bp * binder_res_mask.unsqueeze(-1)).sum(dim=1)  # [B, 3]
            binder_len = binder_res_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            binder_prop = binder_counts / binder_len  # [B, 3], rows sum ~= 1
            # Quantize to 0.1 bins.
            binder_prop_q = (binder_prop * 10.0).round() / 10.0
            if is_unindexed:
                # Apply across all valid protein-body positions (model does not know
                # motif locations under unindexed conditioning), then pad zeros at
                # the appended motif positions.
                binder_sse_prop_pb = (
                    binder_prop_q.unsqueeze(1) * seq_mask_pb.unsqueeze(-1)
                )  # [B, N, 3]
                binder_sse_prop_cond = torch.cat(
                    [
                        binder_sse_prop_pb,
                        torch.zeros(bs, M, 3, device=device, dtype=binder_sse_prop_pb.dtype),
                    ],
                    dim=1,
                )
            else:
                # Indexed: broadcast across binder (non-motif) positions only.
                binder_sse_prop_cond = (
                    binder_prop_q.unsqueeze(1) * binder_res_mask.unsqueeze(-1)
                )  # [B, N, 3]
            binder_sse_prop_cond = binder_sse_prop_cond * keep_b

        # Per-residue motif SASA burial-class conditioning (cc91-sasa). Place the
        # buried/partial/exposed one-hot from `sasa_class` at the motif residue
        # positions, sharing the same `keep_b` CFG-dropout draw as the SSE channels.
        # Mirrors motif_sse_cond: indexed -> [B, N, 3] at motif positions; unindexed
        # -> [B, N+M, 3] with the labels on the appended motif tail.
        motif_sasa_cond = None
        if (
            self.config.train.crop_conditional
            and "sasa" in self.config.model.conditioning_style
            and "sasa_class" in inputs
        ):
            sasa_oh = inputs["sasa_class"].float()  # [B, N, 3]
            if is_unindexed:
                motif_sasa_at_motif = torch.zeros(
                    bs, M, 3, device=device, dtype=sasa_oh.dtype
                )
                for bi in range(bs):
                    motif_idx = torch.nonzero(motif_res_mask[bi]).squeeze(-1)
                    n_m = min(len(motif_idx), M)
                    if n_m > 0:
                        motif_sasa_at_motif[bi, :n_m] = sasa_oh[bi, motif_idx[:n_m]]
                motif_sasa_at_motif = motif_sasa_at_motif * keep_b
                motif_sasa_cond = torch.cat(
                    [
                        torch.zeros(bs, protein_N, 3, device=device, dtype=sasa_oh.dtype),
                        motif_sasa_at_motif,
                    ],
                    dim=1,
                )
            else:
                motif_res_mask_indexed = (
                    crop_cond_mask.any(-1).float() if crop_cond_mask is not None else None
                )
                if motif_res_mask_indexed is not None:
                    motif_sasa_cond = sasa_oh * motif_res_mask_indexed.unsqueeze(-1)
                else:
                    motif_sasa_cond = torch.zeros_like(sasa_oh)
                motif_sasa_cond = motif_sasa_cond * keep_b

        # Per-atom motif h-bond conditioning (cc91-sasa-hbond). For each example,
        # pull the side-chain h-bonds that TOUCH the subsampled motif (>=1 partner
        # atom at a motif residue), then drop each independently with a per-bond-
        # TYPE probability: `drop_prob_bb_sc` for backbone<->side-chain bonds and
        # `drop_prob_sc_sc` for side-chain<->side-chain bonds. Mark the motif-side
        # atom(s) of the surviving bonds in a per-atom [B, N, 37] channel. Shown
        # for a configurable fraction of examples (`cond_prob`, default 1.0);
        # independent of the SSE/SASA keep_b. Indexed conditioning only.
        motif_hbond_cond = None
        if (
            self.config.train.crop_conditional
            and "hbond" in self.config.model.conditioning_style
            and "hbond_pairs" in inputs
            and not is_unindexed
            and crop_cond_mask is not None
        ):
            hb_cfg = getattr(self.config.train, "hbond_cond", None)
            cond_prob = float(getattr(hb_cfg, "cond_prob", 1.0)) if hb_cfg else 1.0
            drop_bb_sc = float(getattr(hb_cfg, "drop_prob_bb_sc", 0.9)) if hb_cfg else 0.9
            drop_sc_sc = float(getattr(hb_cfg, "drop_prob_sc_sc", 0.5)) if hb_cfg else 0.5
            n_atoms_ch = crop_cond_mask.shape[-1]
            bb_polar = (0, 4, 36)  # backbone polar atoms in atom37: N, O, OXT
            hbp = inputs["hbond_pairs"].cpu().numpy()  # [B, H_max, 4], -1 padded
            motif_bool = crop_cond_mask.any(-1).cpu().numpy()  # [B, N] bool
            N = motif_bool.shape[1]
            chan = np.zeros((bs, N, n_atoms_ch), dtype=np.float32)
            shows = np.random.rand(bs) < cond_prob
            for bi in range(bs):
                if not shows[bi]:
                    continue
                pairs = hbp[bi]
                pairs = pairs[pairs[:, 0] >= 0]  # drop padding rows
                if len(pairs) == 0:
                    continue
                rD, aD, rA, aA = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]
                inb = (rD < N) & (rA < N)  # ignore residues past the crop window
                rD, aD, rA, aA = rD[inb], aD[inb], rA[inb], aA[inb]
                if len(rD) == 0:
                    continue
                mset = motif_bool[bi]
                touch = mset[rD] | mset[rA]  # bonds involving the motif
                idx = np.nonzero(touch)[0]
                if len(idx) == 0:
                    continue
                # Classify each touching bond: exactly one backbone atom -> bb-sc,
                # else sc-sc (bb-bb was excluded at precompute).
                is_bb_D = np.isin(aD, bb_polar)
                is_bb_A = np.isin(aA, bb_polar)
                is_bb_sc = is_bb_D ^ is_bb_A
                p_drop = np.where(is_bb_sc, drop_bb_sc, drop_sc_sc)
                keep = idx[np.random.rand(len(idx)) >= p_drop[idx]]
                for j in keep:
                    if mset[rD[j]]:
                        chan[bi, rD[j], aD[j]] = 1.0
                    if mset[rA[j]]:
                        chan[bi, rA[j], aA[j]] = 1.0
            motif_hbond_cond = torch.from_numpy(chan).to(
                device=device, dtype=coords.dtype
            )

        # Noise data
        # Sample time
        is_flow_matching = getattr(self.config.model, "objective", "denoising") == "flow_matching"
        if timestep is None:
            timestep = torch.rand(bs).clamp(min=1e-9, max=1 - 1e-9).to(self.device)

        protein_coords_src = protein_coords if is_unindexed else coords
        atom_mask_src = protein_atom_mask if is_unindexed else atom_mask
        noise_epsilon = None

        if is_flow_matching:
            # Flow matching: t directly, linear interpolation
            noise_level = timestep  # pass t as noise_level to model
            noised_coords, noise_epsilon = diffusion.flow_matching_interpolate(
                protein_coords_src,
                noise_level,
                atom_mask=atom_mask_src,
                dummy_fill_mode=self.config.data.dummy_fill_mode,
            )
        else:
            noise_level = self.model.training_noise_schedule(timestep)
            noised_coords = diffusion.noise_coords(
                protein_coords_src,
                noise_level,
                atom_mask=atom_mask_src,
                dummy_fill_mode=self.config.data.dummy_fill_mode,
            )

        bb_seq = ((protein_seq_mask if is_unindexed else seq_mask) * residue_constants.restype_order["G"]).long()
        bb_atom_mask = atom37_mask_from_aatype(bb_seq, protein_seq_mask if is_unindexed else seq_mask)

        # some backbone atoms may be missing -- mask them to zeros!
        _am = protein_atom_mask if is_unindexed else atom_mask
        if bb_atom_mask.shape[-1] < _am.shape[-1]:
            # cc152 uses 37 canonical slots + an extra-atom block; the backbone
            # mask only ever covers the canonical slots, so pad the rest false.
            bb_atom_mask = F.pad(
                bb_atom_mask, (0, _am.shape[-1] - bb_atom_mask.shape[-1])
            )
        bb_atom_mask = torch.logical_and(bb_atom_mask, _am)

        if self.config.model.task == "backbone":
            noised_coords = noised_coords * bb_atom_mask[..., None]
        elif self.config.model.task == "ai-allatom":
            noised_coords = noised_coords * (protein_atom_mask if is_unindexed else atom_mask)[..., None]

        # cc152 vs cc153: how ligand / cofactor / metal rows are treated.
        #   "denoised"          (cc153) -- noised and supervised like protein.
        #   "conditioning_only" (cc152) -- handed to the model clean at every
        #       noise level and excluded from the loss, so the ligand is fixed,
        #       known context that the protein is generated around. This is what
        #       an ensemble-fitting caller has: the crystallographic ligand is
        #       given, the protein conformation is the unknown.
        ligand_treatment = getattr(self.config.model, "ligand_treatment", "denoised")
        ligand_row = inputs.get("is_ligand")
        ligand_atom_mask = None
        if (
            ligand_treatment == "conditioning_only"
            and ligand_row is not None
            and not is_unindexed
        ):
            # [B, N, 1] -> atoms of ligand rows that actually exist
            ligand_atom_mask = (ligand_row.unsqueeze(-1) > 0) & (atom_mask > 0)
            keep = ligand_atom_mask.unsqueeze(-1)
            noised_coords = torch.where(keep, coords, noised_coords)
            if crop_cond_mask is not None:
                # also surface them through the explicit "this is given" channel
                crop_cond_mask = torch.clamp(
                    crop_cond_mask + ligand_atom_mask.to(crop_cond_mask.dtype), max=1.0
                )
                struct_crop_cond = coords * crop_cond_mask[..., None]

        # For unindexed: concatenate clean motif coords after noised protein
        if is_unindexed:
            noised_coords = torch.cat([noised_coords, motif_clean_coords], dim=1)
            residue_index = torch.cat([
                inputs["residue_index"],
                torch.zeros(bs, M, device=device, dtype=inputs["residue_index"].dtype),
            ], dim=1)
            chain_index = torch.cat([
                inputs["chain_index"],
                torch.zeros(bs, M, device=device, dtype=inputs["chain_index"].dtype),
            ], dim=1)
        else:
            residue_index = inputs["residue_index"]
            chain_index = inputs["chain_index"]

        # Forward pass
        model_inputs = {
            "noisy_coords": noised_coords,
            "noise_level": noise_level,
            "seq_mask": seq_mask,
            "residue_index": residue_index,
            "chain_index": chain_index,
            "hotspot_mask": (
                hotspot_mask
                if "hotspot" in self.config.model.conditioning_style
                else None
            ),
            "struct_crop_cond": struct_crop_cond,
            "crop_cond_mask": (
                crop_cond_mask
                if self.config.train.crop_conditional
                and "condmask" in self.config.model.conditioning_style
                else None
            ),
            "sse_cond": sse_cond,
            "adj_cond": adj_cond,
            "motif_sse_cond": motif_sse_cond,
            "binder_sse_prop_cond": binder_sse_prop_cond,
            "motif_sasa_cond": motif_sasa_cond,
            "motif_hbond_cond": motif_hbond_cond,
            # Atomized per-atom slot descriptors (cc152). Present only for the
            # atomized datasets; absent -> the denoiser fills zeros.
            "atom_elem": inputs.get("atom_elem"),
            "atom_ref": inputs.get("atom_ref"),
            "atom_noncanon": inputs.get("atom_noncanon"),
            "atom_bondfeat": inputs.get("atom_bondfeat"),
        }

        forward_fn = self.forward if is_training else self.model
        struct_self_cond, seq_self_cond = None, None

        if hasattr(self.config.model, "debug_mpnn") and self.config.model.debug_mpnn:
            if (
                np.random.uniform() < self.config.train.self_cond_train_prob
                and self.config.model.mpnn_model.use_self_conditioning
            ):
                with torch.no_grad():
                    _, _, _, seq_self_cond = forward_fn(
                        **model_inputs,
                    )
            _, pred_seq_logprobs, _, _ = forward_fn(
                **model_inputs,
                seq_self_cond=seq_self_cond,
            )
        else:
            if np.random.uniform() < self.config.train.self_cond_train_prob:
                with torch.no_grad():
                    _, _, struct_self_cond, seq_self_cond = forward_fn(**model_inputs)
                # Override motif self-cond with clean coords / sigma_data
                if is_unindexed and struct_self_cond is not None:
                    struct_self_cond[:, protein_N:] = motif_clean_coords / self.sigma_data
            denoised_coords, pred_seq_logprobs, _, _ = forward_fn(
                **model_inputs,
                struct_self_cond=struct_self_cond,
                seq_self_cond=seq_self_cond,
            )

        # For unindexed: extract only protein portion for loss
        if is_unindexed:
            model_output = denoised_coords[:, :protein_N]
            if pred_seq_logprobs.shape[1] > protein_N:
                pred_seq_logprobs = pred_seq_logprobs[:, :protein_N]
            coords = protein_coords
            seq_mask = protein_seq_mask
            atom_mask = protein_atom_mask
        else:
            model_output = denoised_coords

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        aux = {}

        # Compute structure loss
        if self.config.model.task in [
            "backbone",
            "allatom",
            "ai-allatom",
            "ai-allatom-nomask",
            "codesign",
        ]:
            if self.config.model.task == "backbone":
                struct_loss_mask = torch.ones_like(coords) * bb_atom_mask[..., None]
            else:
                if self.config.model.compute_loss_on_all_atoms:
                    # Compute loss on all 37 atoms
                    struct_loss_mask = torch.ones_like(
                        coords
                    ) * unsqueeze_trailing_dims(seq_mask, coords)
                else:
                    struct_loss_mask = torch.ones_like(coords) * atom_mask[..., None]

            # cc152: ligands are context, not a prediction target.
            if ligand_atom_mask is not None:
                struct_loss_mask = struct_loss_mask * (
                    ~ligand_atom_mask
                ).unsqueeze(-1).to(struct_loss_mask.dtype)

            if is_flow_matching:
                # Flow matching: velocity loss with uniform weighting
                velocity_target = coords - noise_epsilon  # x_0 - ε
                struct_loss = masked_mse(
                    velocity_target, model_output, struct_loss_mask
                )
            else:
                # Denoising: x_0 loss with Karras weighting
                loss_weight = (noise_level**2 + self.sigma_data**2) / (
                    (noise_level * self.sigma_data) ** 2
                )
                struct_loss = masked_mse(
                    coords, model_output, struct_loss_mask, loss_weight
                )
            loss = loss + struct_loss
            aux["struct_loss"] = struct_loss.mean().detach().cpu().item()

            # cc154: explicit bonded-distance penalty. Per-atom MSE cannot
            # distinguish "both partners displaced together" (bond intact) from
            # "displaced oppositely" (bond destroyed), which is why the
            # extra-atom block reached 70% broken bonds while canonical slots
            # stayed at 0.4%. Applied to the predicted x0 over the CCD bond
            # list, only where both partners are in the structure loss (so
            # cc152-style clamped ligands contribute nothing).
            bond_w = float(getattr(self.config.train, "bond_loss_weight", 0.0))
            bond_pairs = inputs.get("bond_pairs")
            if bond_w > 0.0 and bond_pairs is not None:
                warm = int(getattr(self.config.train, "bond_loss_warmup", 0))
                step = getattr(self, "_bond_step", 0)
                if is_training:
                    self._bond_step = step + 1
                scale = bond_w * (min(1.0, (step + 1) / warm) if warm > 0 else 1.0)

                target_len = inputs["bond_lengths"]
                n_tok = model_output.shape[1]
                ti = bond_pairs[..., 0]
                si = bond_pairs[..., 1]
                tj = bond_pairs[..., 2]
                sj = bond_pairs[..., 3]
                valid = (ti >= 0) & (ti < n_tok) & (tj >= 0) & (tj < n_tok)
                tic, sic = ti.clamp(min=0), si.clamp(min=0)
                tjc, sjc = tj.clamp(min=0), sj.clamp(min=0)
                bidx = torch.arange(bs, device=device).unsqueeze(-1)
                xi = model_output[bidx, tic, sic]
                xj = model_output[bidx, tjc, sjc]
                # both partners must actually be supervised
                mi = struct_loss_mask[bidx, tic, sic, 0] > 0
                mj = struct_loss_mask[bidx, tjc, sjc, 0] > 0
                valid = valid & mi & mj
                d_pred = (xi - xj).norm(dim=-1)
                sq = (d_pred - target_len) ** 2 * valid.float()
                bond_loss = sq.sum(-1) / valid.float().sum(-1).clamp(min=1.0)
                loss = loss + scale * bond_loss
                aux["bond_loss"] = bond_loss.mean().detach().cpu().item()
                aux["bond_loss_scale"] = scale
                aux["bond_pairs_used"] = (
                    valid.float().sum(-1).mean().detach().cpu().item()
                )

        # Compute mpnn loss
        if self.config.model.task in ["seqdes", "codesign"]:
            alpha = self.config.model.mpnn_model.label_smoothing
            target_oh = (1 - alpha) * aatype_oh + alpha / self.model.n_tokens
            seq_loss_mask = seq_mask
            mpnn_loss = masked_cross_entropy(
                pred_seq_logprobs, target_oh, seq_loss_mask
            )
            loss = loss + mpnn_loss
            aux["mpnn_loss"] = mpnn_loss.mean().detach().cpu().item()

        aux["train_loss"] = loss.mean().detach().cpu().item()
        if return_aux:
            return loss.mean(), aux
        return loss.mean()

    def train_step(self, inputs) -> dict[str, float]:
        self.model.zero_grad()

        if self.scaler is not None:
            with autocast("cuda"):
                loss, log_dict = self.compute_loss(
                    inputs, is_training=True, return_aux=True
                )
                self.scaler.scale(loss).backward()

                # Compute the gradient norm and add it to the log_dict
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), float("inf")
                )
                log_dict["grad_norm"] = grad_norm.item()

                try:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip_val
                    )
                except RuntimeError as e:
                    print(f"Warning: Failed to clip gradients: {e}")

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            loss, log_dict = self.compute_loss(
                inputs, is_training=True, return_aux=True
            )
            loss.backward()

            # Compute the gradient norm and add it to the log_dict
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
            log_dict["grad_norm"] = grad_norm.item()

            try:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip_val
                )
            except RuntimeError as e:
                print(f"Warning: Failed to clip gradients: {e}")
            self.optimizer.step()
            self.scheduler.step()

        # Add train prefix to all keys
        keys = list(log_dict.keys())
        for k in keys:
            log_dict[f"train/{k}"] = log_dict.pop(k)

        return log_dict


@record
def _expand_env_in_config(obj):
    """Back-compat alias for :func:`protpardelle.utils.expand_env_in_config`.

    Kept because the release package's verify_training.sh imports this name.
    """

    return expand_env_in_config(obj)


def train(
    config_path: str,
    out_dir: str,
    project: str,
    wandb_id: str = "",
    exp_name: str | None = None,
    overfit: int = -1,
    debug: bool = False,
    gpu_id: int = 0,
    use_dataparallel: bool = False,
    use_ddp: bool = False,
    num_workers: int = 0,
) -> None:

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config_dict = _expand_env_in_config(config_dict)
    config = dict_to_namespace(config_dict)

    if debug:
        print("Debug mode is enabled")
        num_workers = 0  # Override num_workers to 0 in debug mode

    # DDP setup
    if use_ddp:
        # Longer NCCL collective timeout (default 10 min) so a transient data-loading
        # / Lustre IO stall on one rank doesn't trip the watchdog and abort the run.
        import datetime as _dt
        dist.init_process_group(backend="nccl", timeout=_dt.timedelta(minutes=60))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = local_rank == 0
    else:
        is_main = True

    # Set seeds
    seed = config.train.seed
    if seed is not None:
        seed_everything(seed)
    # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.deterministic = True
    # nonrandom selection of CUDNN convolution, maybe slower
    torch.backends.cudnn.benchmark = False

    wandb_dir = str(Path(out_dir, project))
    if is_main:
        Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)  # Create wandb dir

    # Set up devices
    if not use_ddp:
        if torch.cuda.is_available():
            torch.cuda.init()
            print("Device count:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device())
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

    # Load in datasets, using torch concatenation to combine datasets if there are multiple specified
    # assumes the first dataset is the main dataset for computing epochs, sample from the other datasets with replacement
    teddymer_filters = getattr(config.data, "teddymer_filters", None)
    if teddymer_filters is not None and not isinstance(teddymer_filters, dict):
        teddymer_filters = vars(teddymer_filters)

    # dssp_csv may be a single path (shared across datasets) or a list aligned
    # with pdb_paths (one per dataset).
    _dssp_cfg = getattr(config.data, "dssp_csv", None)
    def _dssp_for(di: int):
        if _dssp_cfg is None:
            return None
        if isinstance(_dssp_cfg, (list, tuple)):
            return _dssp_cfg[di] if di < len(_dssp_cfg) else None
        return _dssp_cfg

    # sasa_csv follows the same single-path-or-list convention as dssp_csv.
    _sasa_cfg = getattr(config.data, "sasa_csv", None)
    def _sasa_for(di: int):
        if _sasa_cfg is None:
            return None
        if isinstance(_sasa_cfg, (list, tuple)):
            return _sasa_cfg[di] if di < len(_sasa_cfg) else None
        return _sasa_cfg

    _sasa_thresholds = getattr(config.data, "sasa_thresholds", None)

    # hbond_csv follows the same single-path-or-list convention as dssp/sasa.
    _hbond_cfg = getattr(config.data, "hbond_csv", None)
    def _hbond_for(di: int):
        if _hbond_cfg is None:
            return None
        if isinstance(_hbond_cfg, (list, tuple)):
            return _hbond_cfg[di] if di < len(_hbond_cfg) else None
        return _hbond_cfg

    _hbond_max = int(getattr(config.data, "hbond_max", 384))

    def _as_dict(node):
        """Config sub-tree -> plain dict (configs arrive as argparse.Namespace)."""

        if node is None:
            return None
        if isinstance(node, dict):
            return dict(node)
        return dict(vars(node))

    train_datasets = [
        PDBDataset(
            pdb_path=config.data.pdb_paths[di],
            fixed_size=config.data.fixed_size,
            mode="train",
            overfit=overfit,
            short_epoch=debug,
            se3_data_augment=config.data.se3_data_augment,
            translation_scale=config.data.translation_scale,
            chain_residx_gap=config.data.chain_residx_gap,
            dummy_fill_mode=config.data.dummy_fill_mode,
            subset=config.data.subset[di],
            dssp_csv=_dssp_for(di),
            sasa_csv=_sasa_for(di),
            sasa_thresholds=_sasa_thresholds,
            hbond_csv=_hbond_for(di),
            hbond_max=_hbond_max,
            metadata_tsv=getattr(config.data, "metadata_tsv", None),
            teddymer_filters=teddymer_filters,
            long_chain_spatial_crop=getattr(
                config.data, "long_chain_spatial_crop", False
            ),
            long_chain_max_len=getattr(config.data, "long_chain_max_len", 256),
            mirror_path=getattr(config.data, "mirror_path", None),
            ccd_cache=getattr(config.data, "ccd_cache", None),
            atomize_kwargs=_as_dict(getattr(config.data, "atomize", None)),
        )
        for di in range(len(config.data.pdb_paths))
    ]
    dataset = ConcatDataset(train_datasets)

    if use_ddp:
        train_sampler = DistributedSampler(dataset, shuffle=True)
    else:
        train_sampler = StochasticMixedSampler(
            train_datasets, config.data.mixing_ratios, batch_size=config.train.batch_size
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=num_workers,
        pin_memory="cuda" in device,
        shuffle=False,  # the sampler takes care of shuffling
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=not debug,
    )

    # Calculate sigma_data
    if config.data.auto_calc_sigma_data:
        print(
            f"===== Automatically computing sigma_data for {config.data.n_examples_for_sigma_data} examples ====="
        )
        sigma_data = calc_sigma_data(dataset, config, num_workers)

        # Update config and config_dict with estimated sigma_data
        config.data.sigma_data = sigma_data
        config_dict["data"]["sigma_data"] = sigma_data

    # Init wandb and logging for process 0
    log_dir = ""
    wandb.init(
        mode="disabled" if (debug or not is_main) else "online",
        project=project,
        entity=wandb_id,
        name=exp_name,
        job_type="debug" if debug else "train",
        config=config_dict,
        dir=wandb_dir,
    )
    if is_main:
        if wandb.run:
            print(
                f"Beginning: run_name={wandb.run.name}, run_id={wandb.run.id}, device={device}"
            )
        else:
            print(f"Beginning: device={device}")
        print(f"Training configuration: {config_path=}, {out_dir=}, {project=}")

    # Set up logging
    run_name = "default_run"
    if wandb.run and wandb.run.name:
        run_name = wandb.run.name
    if use_ddp and not is_main:
        run_name = exp_name or "default_run"

    log_dir = Path(out_dir, project, f"{run_name}_debug" if debug else run_name)

    if is_main:
        Path(log_dir, "results").mkdir(parents=True, exist_ok=True)
        Path(log_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

        # Preserve config
        with open(Path(log_dir, "config.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f)

    # Set up model and optimizers
    model = models.Protpardelle(config, device)
    start_epoch, total_steps = 0, 0
    if config.train.ckpt_path != "":
        training_state = torch.load(
            config.train.ckpt_path, weights_only=False, map_location=device
        )
        model.load_state_dict(training_state["model_state_dict"])
        start_epoch = training_state["epoch"]
        total_steps = training_state["total_steps"]

    # Optionally build a frozen teacher for FastGen-style distillation.
    teacher_model = None
    distill_cfg = getattr(config.train, "distill", None)
    if distill_cfg is not None and getattr(distill_cfg, "enabled", False):
        teacher_path = distill_cfg.teacher_ckpt_path
        if not teacher_path:
            raise ValueError(
                "train.distill.enabled=True but train.distill.teacher_ckpt_path is empty"
            )
        if is_main:
            print(f"===== Loading distillation teacher from {teacher_path} =====")
        teacher_state = torch.load(
            teacher_path, weights_only=False, map_location=device
        )
        teacher_model = models.Protpardelle(config, device)
        if isinstance(teacher_state, dict) and "model_state_dict" in teacher_state:
            teacher_model.load_state_dict(teacher_state["model_state_dict"])
        else:
            teacher_model.load_state_dict(teacher_state.state_dict())
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.to(device)
        # Initialize student from teacher when no student ckpt is provided.
        if config.train.ckpt_path == "" and getattr(
            distill_cfg, "init_student_from_teacher", True
        ):
            if is_main:
                print("===== Initializing student from teacher weights =====")
            if isinstance(teacher_state, dict) and "model_state_dict" in teacher_state:
                model.load_state_dict(teacher_state["model_state_dict"])
            else:
                model.load_state_dict(teacher_state.state_dict())

    if use_ddp:
        model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    elif use_dataparallel:
        model = nn.DataParallel(model)

    model.train()
    if not use_ddp:
        model.to(device)

    runner = ProtpardelleRunner(
        config,
        model,
        dataset,
        log_dir,
        device,
        teacher_model=teacher_model,
    )
    if (
        config.train.ckpt_path != ""
    ):
        for _ in range(total_steps):
            runner.scheduler.step()
        runner.optimizer.load_state_dict(training_state["optim_state_dict"])
    runner.train_init()

    # Post-hoc EMA setup
    phema_cfg = getattr(config.train, "phema", None)
    ema_tracker = None
    if phema_cfg is not None and getattr(phema_cfg, "enabled", False):
        phema_stds = list(phema_cfg.stds)
        ema_tracker = PowerFunctionEMA(runner.model, stds=phema_stds)
        if config.train.ckpt_path != "" and "phema_state" in training_state:
            ema_tracker.load_state_dict(training_state["phema_state"])
            print(f"Loaded post-hoc EMA state (stds={phema_stds})")
        else:
            print(f"Initialized post-hoc EMA tracking (stds={phema_stds})")

    with torch.autograd.set_detect_anomaly(True) if debug else nullcontext():
        for epoch in range(start_epoch + 1, config.train.max_epochs + 1):
            if use_ddp:
                train_sampler.set_epoch(epoch)
            for inputs in tqdm(
                dataloader,
                desc=f"epoch {epoch}/{config.train.max_epochs}",
                disable=not is_main,
            ):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["step"] = total_steps
                log_dict = runner.train_step(inputs)
                log_dict["learning_rate"] = runner.scheduler.get_last_lr()[0]

                total_steps += 1

                # Update post-hoc EMA
                if ema_tracker is not None:
                    ema_tracker.update(
                        cur_nimg=total_steps * config.train.batch_size,
                        batch_size=config.train.batch_size,
                    )

                if is_main:
                    wandb.log(log_dict, step=total_steps)
                    wandb.log({"epoch": epoch}, step=total_steps)

            with torch.no_grad():  # per epoch
                # Save checkpoint
                if is_main and (
                    epoch % config.train.checkpoint_freq == 0
                    or epoch in config.train.checkpoints
                ):
                    runner.model.eval()
                    torch.save(
                        runner.model,
                        f"{log_dir}/checkpoints/epoch{epoch}_model.pth",
                    )
                    ckpt_dict = {
                        "model_state_dict": runner.model.state_dict(),
                        "optim_state_dict": runner.optimizer.state_dict(),
                        "epoch": epoch,
                        "total_steps": total_steps,
                    }
                    if ema_tracker is not None:
                        ckpt_dict["phema_state"] = ema_tracker.state_dict()
                    torch.save(
                        ckpt_dict,
                        f"{log_dir}/checkpoints/epoch{epoch}_training_state.pth",
                    )

                    # Save per-profile EMA snapshots for post-hoc reconstruction
                    if ema_tracker is not None:
                        for ema_net, ema_suffix in ema_tracker.get():
                            torch.save(
                                ema_net.state_dict(),
                                f"{log_dir}/checkpoints/epoch{epoch}_ema{ema_suffix}.pth",
                            )

                    runner.model.train()

    if is_main:
        wandb.finish()
        if wandb.run and wandb.run.dir:
            subprocess.run(shlex.split(f"cp -r {wandb.run.dir} {log_dir}"), check=False)
        if wandb.run:
            print(
                f'Training finished. (run name "{wandb.run.name}", run id "{wandb.run.id}")'
            )
        else:
            print("Training finished.")

    if use_ddp:
        dist.destroy_process_group()


@app.command()
def main(
    project: str = typer.Option("other", help="wandb project name"),
    wandb_id: str = typer.Option("", help="wandb username"),
    exp_name: str = typer.Option(None, help="wandb exp name"),
    config: str = typer.Option("configs/config.yml", help="experiment config"),
    overfit: int = typer.Option(-1, help="number of examples to overfit to"),
    debug: bool = typer.Option(False, help="run one batch and eval offline"),
    gpu_id: int = typer.Option(0, help="which GPU to use"),
    use_dataparallel: bool = typer.Option(False, help="use DataParallel"),
    use_ddp: bool = typer.Option(False, help="use DistributedDataParallel (launch with torchrun)"),
    num_workers: int = typer.Option(0, help="dataloader num workers"),
    out_dir: str = typer.Option(..., help="output path for trained models"),
):

    train(
        config_path=config,
        out_dir=out_dir,
        project=project,
        wandb_id=wandb_id,
        exp_name=exp_name,
        overfit=overfit,
        debug=debug,
        gpu_id=gpu_id,
        use_dataparallel=use_dataparallel,
        use_ddp=use_ddp,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    app()
