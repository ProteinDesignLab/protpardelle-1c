"""Entrypoint for Protpardelle-1c training.

Authors: Alex Chu, Zhaoyang Li, Richard Shuai, Tianyu Lu
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import typer
import wandb
import yaml
from torch.amp import GradScaler, autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.types import Device
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.configs import TrainingConfig
from protpardelle.core import modules
from protpardelle.core.models import Protpardelle
from protpardelle.data.atom import atom37_mask_from_aatype, dummy_fill_noise_coords
from protpardelle.data.dataset import (
    PDBDataset,
    StochasticMixedSampler,
    calc_sigma_data,
    make_crop_cond_mask_and_recenter_coords,
)
from protpardelle.utils import (
    StrPath,
    enable_tf32_if_available,
    get_default_device,
    get_logger,
    load_config,
    namespace_to_dict,
    norm_path,
    seed_everything,
    unsqueeze_trailing_dims,
)

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@dataclass
class DistributedContext:
    """Container for distributed training metadata."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """Return True if the current process is the primary (rank 0)."""

        return self.rank == 0

    @property
    def ddp_enabled(self) -> bool:
        """Return True if distributed data parallel is enabled."""
        return self.world_size > 1

    @classmethod
    def empty_context(cls) -> DistributedContext:
        """Return an empty distributed context."""

        return cls(rank=0, local_rank=0, world_size=1)


def _resolve_device_with_distributed(
    requested_device: torch.device,
) -> tuple[torch.device, DistributedContext]:
    """Initialize torch.distributed if launched with multiple processes.

    Args:
        requested_device (torch.device): Optional device requested via CLI.

    Returns:
        tuple[torch.device, DistributedContext]: Possibly updated device and
            the distributed context when multi-process training is active.
    """

    if not dist.is_available():
        return requested_device, DistributedContext.empty_context()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return requested_device, DistributedContext.empty_context()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        resolved_device = torch.device(f"cuda:{local_rank}")
    else:
        resolved_device = torch.device("cpu")

    if requested_device != resolved_device:
        logger.warning(
            "Overriding requested device %s with local rank device %s for DDP.",
            requested_device,
            resolved_device,
        )

    return resolved_device, DistributedContext(
        rank=rank, local_rank=local_rank, world_size=world_size
    )


def _cleanup_distributed(distributed: DistributedContext) -> None:
    """Tear down the distributed process group if it was initialized."""

    if not distributed.ddp_enabled:
        return
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except RuntimeError as e:  # Some ranks may have crashed already.
            logger.debug("Skipping distributed barrier during cleanup: %s", e)
        try:
            dist.destroy_process_group()
        except RuntimeError as e:
            logger.debug("Failed to destroy process group cleanly: %s", e)


def _log_distributed_mean(
    log_dict: dict[str, float],
    device: torch.device,
    distributed: DistributedContext,
) -> dict[str, float]:
    """Average scalar metrics across distributed processes."""

    if not (distributed.ddp_enabled and log_dict):
        return log_dict

    keys = sorted(log_dict.keys())
    values = torch.tensor([log_dict[k] for k in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.AVG)

    return {k: values[i].item() for i, k in enumerate(keys)}


def masked_cross_entropy_loss(
    logprobs: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute the masked cross-entropy loss.

    Args:
        logprobs (torch.Tensor): Log probabilities of the predicted tokens.
        target (torch.Tensor): One-hot encoded target tokens.
        loss_mask (torch.Tensor): Mask to apply to the loss.
        tol (float, optional): Tolerance for the loss computation. Defaults to 1e-6.

    Returns:
        torch.Tensor: The computed masked cross-entropy loss.
    """

    cel = -target * logprobs
    cel = cel * loss_mask.unsqueeze(-1)
    cel = cel.sum((-1, -2)) / loss_mask.sum(-1).clamp(min=tol)

    return cel


def masked_mse_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute the masked mean squared error loss.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values.
        mask (torch.Tensor): Mask to apply to the loss.
        weights (torch.Tensor | None, optional): Weights to apply to the loss. Defaults to None.
        tol (float, optional): Tolerance for the loss computation. Defaults to 1e-6.

    Returns:
        torch.Tensor: The computed masked mean squared error loss.
    """

    data_dims = tuple(range(1, x.ndim))
    mse = (x - y).square() * mask
    if weights is not None:
        mse = mse * unsqueeze_trailing_dims(weights, mse)
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=tol)

    return mse


def load_datasets(config: TrainingConfig) -> list[PDBDataset]:
    """Load the training datasets."""
    datasets = [
        PDBDataset(
            pdb_path=pdb_path,
            subset=subset,
            fixed_size=config.data.fixed_size,
            short_epoch=config.data.short_epoch,
            se3_data_augment=config.data.se3_data_augment,
            translation_scale=config.data.translation_scale,
            chain_residx_gap=config.data.chain_residx_gap,
            dummy_fill_mode=config.data.dummy_fill_mode,
        )
        for pdb_path, subset in zip(config.data.pdb_paths, config.data.subset)
    ]

    return datasets


class ProtpardelleTrainer:
    """Trainer for the Protpardelle model."""

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        distributed: DistributedContext,
        batch_size_override: int | None = None,
        num_workers_override: int | None = None,
    ) -> None:
        """Initialize the ProtpardelleTrainer.

        Args:
            config (TrainingConfig): The training configuration.
            device (torch.device): The device to use for training.
            distributed (DistributedContext): Metadata about the distributed setup.
            batch_size_override (int | None, optional): Per-process batch size for distributed
                training. Defaults to None.
            num_workers_override (int | None, optional): Number of dataloader workers.
                Defaults to None.
        """

        # Store config
        self.config = config

        # Store distributed context
        self.distributed = distributed
        self.is_main = self.distributed.is_main
        self.ddp_enabled = self.distributed.ddp_enabled

        # Determine batch size and num_workers
        self.batch_size = (
            batch_size_override
            if batch_size_override is not None
            else self.config.train.batch_size
        )
        self.num_workers = (
            num_workers_override
            if num_workers_override is not None
            else self.config.data.num_workers
        )

        # Initialize model
        model = Protpardelle(self.config, device)
        if self.ddp_enabled:
            logger.info(
                "Initialized DDP rank=%d local_rank=%d world_size=%d",
                self.distributed.rank,
                self.distributed.local_rank,
                self.distributed.world_size,
            )
            model.to(device)
            if device.type == "cuda":
                model = DDP(  # type: ignore
                    model,
                    device_ids=[device.index],
                    output_device=device.index,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    find_unused_parameters=False,  # should be False if using static graph
                    static_graph=True,  # the graph is static; speeds up DDP
                )
            else:
                # Fall back to CPU training with DDP
                model = DDP(  # type: ignore
                    model,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                )
        else:
            if torch.cuda.is_available():
                logger.info("Device count: %d", torch.cuda.device_count())
                logger.info("Current device: %d", torch.cuda.current_device())
            model.to(device)
        model.train()
        self.model = model

        # Initialize optimizer, scheduler, and scaler
        self.optimizer, self.scheduler = self._load_optimizer_and_scheduler()
        self.scaler = GradScaler(device=device, enabled=self.config.train.use_amp)  # type: ignore

    @property
    def module(self) -> Protpardelle:
        """Get the underlying Protpardelle model on the fly."""
        parallel_wrappers = (
            nn.DataParallel,
            DDP,
        )
        return (
            self.model.module
            if isinstance(self.model, parallel_wrappers)
            else self.model
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying Protpardelle model."""
        return self.module.device

    def _load_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Adam, modules.LinearWarmupCosineDecay]:
        """Load the optimizer and scheduler.

        Returns:
            tuple[torch.optim.Adam, modules.LinearWarmupCosineDecay]: The optimizer and scheduler.
        """

        if self.module.task == "seqdes":
            params_to_train = [
                p for n, p in self.module.named_parameters() if "struct_model" not in n
            ]
        else:
            params_to_train = [p for _, p in self.module.named_parameters()]

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

    def save_checkpoint(
        self,
        epoch: int,
        total_steps: int,
        checkpoint_dir: StrPath,
    ) -> None:
        """Save the model checkpoint.

        Args:
            epoch (int): The current epoch.
            total_steps (int): The total number of training steps.
            checkpoint_dir (StrPath): The directory to save the checkpoint.
        """

        checkpoint = {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "total_steps": total_steps,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": ".".join(map(str, sys.version_info[:3])),
        }
        checkpoint["rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "sampler_seed": (
                self.config.train.seed if self.config.train.seed is not None else 0
            ),  # only for storing purpose
        }

        checkpoint_dir = norm_path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_dir / f"epoch{epoch}_training_state.pth")

    def load_checkpoint(
        self,
        checkpoint_path: StrPath,
    ) -> tuple[int, int]:
        """Load the model checkpoint.

        Args:
            checkpoint_path (StrPath): The path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
            ValueError: If the checkpoint is invalid.

        Returns:
            tuple[int, int]: The epoch and total steps from the checkpoint.
        """

        checkpoint_path = norm_path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
        if torch.cuda.is_available():
            if checkpoint["rng"]["cuda"] is None:
                raise ValueError(
                    "Checkpoint was trained with CUDA but current device is CPU"
                )
            cuda_states = [state.cpu() for state in checkpoint["rng"]["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(checkpoint["rng"]["numpy"])
        random.setstate(checkpoint["rng"]["python"])

        return checkpoint["epoch"], checkpoint["total_steps"]

    def initialize_training_parameters(self) -> tuple[int, int]:
        """Initialize training parameters.

        Returns:
            tuple[int, int]: The starting epoch and total steps.
        """

        start_epoch = 0
        total_steps = 0

        # Set seeds if no rng provided
        seed = self.config.train.seed
        if seed is not None:
            if self.ddp_enabled:
                seed += self.distributed.rank
            seed_everything(
                seed, freeze_cuda=True
            )  # use deterministic pytorch for training

        return start_epoch, total_steps

    def start_or_resume(self) -> tuple[int, int]:
        """Load checkpoint if it exists, otherwise initialize training parameters.

        Returns:
            tuple[int, int]: The starting epoch and total steps.
        """

        checkpoint_path = self.config.train.ckpt_path
        if checkpoint_path is None:
            return self.initialize_training_parameters()

        checkpoint_path = norm_path(checkpoint_path)
        try:
            start_epoch, total_steps = self.load_checkpoint(checkpoint_path)
            logger.info(
                "Resumed from checkpoint: %s (epoch=%d, total_steps=%d)",
                checkpoint_path,
                start_epoch,
                total_steps,
            )
            return start_epoch, total_steps
        except FileNotFoundError:
            logger.warning(
                "Checkpoint file not found: %s; starting from scratch", checkpoint_path
            )

        return self.initialize_training_parameters()

    def log_training_info(self) -> None:
        """Log training information."""
        logger.info(
            "Total params: %d", sum(p.numel() for p in self.module.parameters())
        )
        logger.info(
            "Trainable params: %d",
            sum(p.numel() for p in self.module.parameters() if p.requires_grad),
        )

    def make_training_collate_fn(
        self,
    ) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        """Create a collate_fn that applies training-time augmentations on CPU."""

        def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            batch_dict = cast(dict[str, torch.Tensor], default_collate(batch))

            if self.config.train.crop_conditional:

                atom_coords = batch_dict["coords_in"]
                atom_mask = batch_dict["atom_mask"]
                aatype = batch_dict["aatype"]
                chain_index = batch_dict["chain_index"]

                # Pre-compute crop conditioning mask and recentered coords for efficiency
                atom_coords, crop_cond_mask, hotspot_mask = (
                    make_crop_cond_mask_and_recenter_coords(
                        atom_coords=atom_coords,
                        atom_mask=atom_mask,
                        aatype=aatype,
                        chain_index=chain_index,
                        **vars(self.config.train.crop_cond),
                    )
                )
                struct_crop_cond = atom_coords * crop_cond_mask.unsqueeze(-1)

                batch_dict["coords_in"] = atom_coords
                batch_dict["crop_cond_mask"] = crop_cond_mask
                batch_dict["struct_crop_cond"] = struct_crop_cond
                batch_dict["hotspot_mask"] = hotspot_mask

            return batch_dict

        return collate_fn

    def get_dataloader(self, datasets: list[PDBDataset]) -> DataLoader:
        """Get the training dataloader.

        Args:
            datasets (list[PDBDataset]): The list of datasets to use.

        Returns:
            DataLoader: The training dataloader.
        """

        # Initialize and combine training datasets. The StochasticMixedSampler will handle
        # sampling from the combined datasets according to specified mixing ratios.

        sampler = StochasticMixedSampler(
            datasets,
            self.config.data.mixing_ratios,
            batch_size=self.batch_size,
            num_replicas=self.distributed.world_size,
            rank=self.distributed.rank,
            seed=self.config.train.seed,
        )

        dataset = ConcatDataset(datasets)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=False,  # the sampler takes care of shuffling
            sampler=sampler,
            drop_last=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.make_training_collate_fn(),
        )

        return dataloader

    def compute_loss(
        self, input_dict: dict[str, torch.Tensor], tol: float = 1e-6
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss for a given input batch.

        Args:
            input_dict (dict[str, torch.Tensor]): Input tensors for the model.
            tol (float, optional): Tolerance for numerical stability. Defaults to 1e-6.

        Raises:
            NotImplementedError: If the all atom loss computation is not implemented.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: The computed loss and logging metrics.
        """

        seq_mask = input_dict["seq_mask"]
        atom_coords = input_dict["coords_in"]
        aatype = input_dict["aatype"]
        atom_mask = input_dict["atom_mask"]
        chain_index = input_dict["chain_index"]
        residue_index = input_dict["residue_index"]

        batch_size = atom_coords.shape[0]

        # Crop conditioning
        if self.config.train.crop_conditional:
            if self.config.model.compute_loss_on_all_atoms:
                raise NotImplementedError(
                    "Crop conditioning with all atom loss not implemented"
                )

            crop_cond_mask = input_dict.get("crop_cond_mask")
            struct_crop_cond = input_dict.get("struct_crop_cond")
            hotspot_mask = input_dict.get("hotspot_mask")

            # If using correct data loader and collate_fn, these should never be None
            assert all(
                x is not None for x in [crop_cond_mask, struct_crop_cond, hotspot_mask]
            )

            if "hotspots" not in self.config.model.conditioning_style:
                hotspot_mask = None  # type: ignore
        else:
            struct_crop_cond = None
            hotspot_mask = None

        # Secondary structure conditioning
        if "ssadj" in self.config.model.conditioning_style:
            sse_cond = input_dict["sse"]
            adj_cond = input_dict["adj"]
        else:
            sse_cond = None
            adj_cond = None

        # Noise data
        timestep = torch.rand(batch_size, device=self.device).clamp(
            min=tol, max=1 - tol
        )
        noise_level = self.module.training_noise_schedule(timestep)
        noised_coords = dummy_fill_noise_coords(
            atom_coords,
            atom_mask,
            noise_level=noise_level,
            dummy_fill_mode=self.config.data.dummy_fill_mode,
        )

        bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
        bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)

        # Some backbone atoms may be missing; mask them to zeros
        bb_atom_mask = (
            bb_atom_mask * atom_mask
        )  # float masks; multiply instead of boolean ops
        if self.config.model.task == "backbone":
            noised_coords = noised_coords * bb_atom_mask.unsqueeze(-1)
        elif self.config.model.task == "ai-allatom":
            noised_coords = noised_coords * atom_mask.unsqueeze(-1)

        # Forward pass
        model_inputs = {
            "noisy_coords": noised_coords,
            "noise_level": noise_level,
            "seq_mask": seq_mask,
            "residue_index": residue_index,
            "chain_index": chain_index,
            "hotspot_mask": hotspot_mask,
            "struct_crop_cond": struct_crop_cond,
            "sse_cond": sse_cond,
            "adj_cond": adj_cond,
        }

        if np.random.rand() < self.config.train.self_cond_train_prob:
            with torch.no_grad():
                _, _, struct_self_cond, seq_self_cond = self.model(**model_inputs)
        else:
            struct_self_cond = None
            seq_self_cond = None
        denoised_coords, aatype_logprobs, _, _ = self.model(
            **model_inputs,
            struct_self_cond=struct_self_cond,
            seq_self_cond=seq_self_cond,
        )

        loss = torch.tensor(0.0, device=self.device)
        log_dict: dict[str, float] = {}

        # Compute structure loss
        if self.config.model.task in {
            "backbone",
            "allatom",
            "ai-allatom",
            "ai-allatom-nomask",
            "codesign",
        }:
            if self.config.model.task == "backbone":
                struct_loss_mask = torch.ones_like(
                    atom_coords
                ) * bb_atom_mask.unsqueeze(-1)
            elif self.config.model.compute_loss_on_all_atoms:
                # Compute loss on all 37 atoms
                struct_loss_mask = torch.ones_like(
                    atom_coords
                ) * unsqueeze_trailing_dims(seq_mask, atom_coords)
            else:
                struct_loss_mask = torch.ones_like(atom_coords) * atom_mask.unsqueeze(
                    -1
                )

            sigma_fp32 = torch.tensor(
                self.config.data.sigma_data,
                device=self.device,
            )
            denom = (noise_level * sigma_fp32).square().clamp(min=tol)
            loss_weight = (noise_level.square() + sigma_fp32.square()) / denom

            struct_loss = masked_mse_loss(
                atom_coords, denoised_coords, struct_loss_mask, loss_weight
            ).mean()
            loss = loss + struct_loss
            log_dict["struct_loss"] = struct_loss.detach().cpu().item()

        # Compute mpnn loss
        if self.config.model.task in {"seqdes", "codesign"}:
            alpha = self.config.model.mpnn_model.label_smoothing
            aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
            target_oh = (1 - alpha) * aatype_oh + alpha / self.module.num_tokens
            mpnn_loss = masked_cross_entropy_loss(
                aatype_logprobs, target_oh, seq_mask
            ).mean()
            loss = loss + mpnn_loss
            log_dict["mpnn_loss"] = mpnn_loss.detach().cpu().item()

        log_dict["train_loss"] = loss.detach().cpu().item()

        return loss, log_dict

    def train_step(self, input_dict: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single training step.

        Args:
            input_dict (dict[str, torch.Tensor]): Input tensors for the model.

        Returns:
            dict[str, float]: Dictionary containing training logging metrics.
        """

        self.optimizer.zero_grad()

        with autocast(self.device.type) if self.config.train.use_amp else nullcontext():
            loss, log_dict = self.compute_loss(input_dict)
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            # Compute the gradient norm and add it to the log_dict
            grad_norm = nn.utils.clip_grad_norm_(
                self.module.parameters(),
                self.config.train.grad_clip_val,
            )
            log_dict["grad_norm"] = grad_norm.item()

            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.scaler.get_scale() >= prev_scale:
            self.scheduler.step()

        # Add train prefix to all keys for wandb logging
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}

        return log_dict


@record
def train(
    config_path: StrPath,
    output_dir: StrPath,
    device: Device = None,
    project_name: str | None = None,
    wandb_id: str | None = None,
    exp_name: str | None = None,
    debug: bool = False,
) -> None:
    """Train a Protpardelle model.

    Args:
        config_path (StrPath): Path to the configuration file.
        output_dir (StrPath): Directory to save output files.
        device (Device, optional): Device to use for training. Defaults to None.
        project_name (str | None, optional): Project name for wandb. Defaults to None.
        wandb_id (str | None, optional): Wandb ID for logging. Defaults to None.
        exp_name (str | None, optional): Experiment name for logging. Defaults to None.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Raises: ...  # TODO
    """

    # Enable TF32 on Ampere+ GPUs for faster training
    tf32_enabled = enable_tf32_if_available()
    if tf32_enabled:
        logger.info("Enabled TF32 mode for faster training on Ampere+ GPUs")

    # Set and resolve device with DDP if applicable
    if device is None:
        requested_device = get_default_device()
    else:
        requested_device = torch.device(device)
    resolved_device, distributed = _resolve_device_with_distributed(requested_device)

    # Load config
    config = load_config(config_path, TrainingConfig)

    # Determine per-process batch size
    global_batch_size = config.train.batch_size
    if distributed.ddp_enabled:
        if global_batch_size % distributed.world_size != 0:
            raise ValueError(
                "train.batch_size must be divisible by the number of distributed processes"
            )
        local_batch_size = global_batch_size // distributed.world_size
        if local_batch_size == 0:
            raise ValueError("Per-process batch size must be at least 1")
    else:
        local_batch_size = global_batch_size

    # Determine number of effective dataloader workers
    global_num_workers = config.data.num_workers
    if debug:
        logger.debug("Debug mode is enabled; setting num_workers to 0")
        global_num_workers = 0
        local_num_workers = 0
    elif distributed.ddp_enabled and (global_num_workers > 0):
        local_num_workers = max(1, global_num_workers // distributed.world_size)
        global_num_workers = local_num_workers * distributed.world_size
    else:
        local_num_workers = global_num_workers

    # Log DDP training setup
    if distributed.ddp_enabled:
        logger.info(
            "Distributed training: rank %d/%d; local/global batch %d/%d; "
            "local/global dataloader workers %d/%d",
            distributed.rank,
            distributed.world_size,
            local_batch_size,
            global_batch_size,
            local_num_workers,
            global_num_workers,
        )
    else:
        logger.info(
            "Single-process training: batch %d, dataloader workers %d",
            global_batch_size,
            global_num_workers,
        )

    # Load datasets
    datasets = load_datasets(config)

    # Auto calculate sigma data if needed
    if config.data.auto_calc_sigma_data:
        sigma_data: float | None = None
        if distributed.is_main:
            dataset = ConcatDataset(datasets)
            sigma_data = calc_sigma_data(dataset, config, num_workers=local_num_workers)
        if distributed.ddp_enabled:
            sigma_tensor = torch.zeros(1, device=resolved_device)
            if distributed.is_main:
                assert sigma_data is not None
                sigma_tensor[0] = sigma_data
            dist.broadcast(sigma_tensor, src=0)
            sigma_data = float(sigma_tensor.item())
        else:
            assert sigma_data is not None

        # Override the config value
        config.data.sigma_data = sigma_data

    # Initialize wandb
    output_dir = norm_path(output_dir)
    wandb_kwargs = {
        "config": config,
        "dir": output_dir,
        "entity": wandb_id,
        "job_type": "debug" if debug else "train",
        "mode": "disabled" if debug else "online",
        "name": exp_name,
        "project": project_name,
    }

    # Disable wandb logging on non-main ranks
    if distributed.ddp_enabled and (not distributed.is_main):
        wandb_kwargs["mode"] = "disabled"

    wandb_run: wandb.Run | None = None
    run_name: str
    run_dir: str | None = None
    run_id: str | None = None

    # Initialize wandb only on the main rank or in debug mode
    if distributed.is_main or debug:
        wandb_run = wandb.init(**wandb_kwargs)
        if wandb_run is None:
            raise RuntimeError("Failed to initialize wandb run")
        if (
            (wandb_run.name is None)
            or (wandb_run.dir is None)
            or (wandb_run.id is None)
        ):
            raise RuntimeError("wandb returned an incomplete run object")
        run_name = wandb_run.name
        run_dir = wandb_run.dir
        run_id = wandb_run.id
    else:
        # Non-main ranks reuse exp_name for logging clarity
        if exp_name is None:
            if distributed.ddp_enabled:
                run_name = f"run-rank{distributed.rank}"
            else:
                run_name = "run"
        else:
            run_name = exp_name

    # Initialize trainer
    trainer = ProtpardelleTrainer(
        config,
        resolved_device,
        distributed,
        batch_size_override=local_batch_size,
        num_workers_override=local_num_workers,
    )

    # Start or resume training from checkpoint
    start_epoch, total_steps = trainer.start_or_resume()

    # Log training info
    if trainer.is_main:
        logger.info(
            "Beginning: run_name %s, run_id %s, device %s",
            run_name,
            run_id,
            trainer.device,
        )
        logger.info(
            "Training configuration: config_path %s, output_dir %s, project_name %s",
            config_path,
            output_dir,
            project_name,
        )

    # Create output directories and save config
    if trainer.is_main:
        log_dir = output_dir / (f"{run_name}_debug" if debug else run_name)
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(namespace_to_dict(trainer.config), f)

        trainer.log_training_info()
    else:
        log_dir = None
        checkpoint_dir = None

    # Initialize dataloader
    dataloader = trainer.get_dataloader(datasets)

    # Wrap the entire training loop in a try-finally to ensure cleanup
    try:
        # Use anomaly detection if in debug mode
        with torch.autograd.set_detect_anomaly(True) if debug else nullcontext():
            for epoch in range(start_epoch + 1, trainer.config.train.max_epochs + 1):
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)  # type: ignore

                progress = tqdm(
                    dataloader,
                    desc=f"epoch {epoch}/{trainer.config.train.max_epochs}",
                    disable=not trainer.is_main,
                )
                for input_dict in progress:
                    input_dict: dict[str, torch.Tensor] = {
                        k: v.to(
                            trainer.device, non_blocking=True
                        )  # non_blocking for pin_memory
                        for k, v in input_dict.items()
                    }
                    log_dict = trainer.train_step(input_dict)
                    log_dict["learning_rate"] = trainer.scheduler.get_last_lr()[0]
                    log_dict["epoch"] = epoch
                    log_dict = _log_distributed_mean(
                        log_dict, trainer.device, distributed
                    )

                    # Log to wandb on main rank only
                    if trainer.is_main:
                        wandb.log(log_dict, step=total_steps)
                    total_steps += 1

                with torch.no_grad():
                    if trainer.is_main:
                        assert checkpoint_dir is not None
                        if (epoch % trainer.config.train.checkpoint_freq == 0) or (
                            epoch in trainer.config.train.checkpoints
                        ):
                            trainer.save_checkpoint(epoch, total_steps, checkpoint_dir)

                if trainer.ddp_enabled:
                    dist.barrier()
    finally:
        if wandb_run is not None:
            wandb.finish()
        if trainer.is_main:
            assert log_dir is not None
            assert run_dir is not None
            # Copy the entire wandb run directory to the log_dir for safekeeping
            subprocess.run(["cp", "-r", str(run_dir), str(log_dir)], check=False)
            logger.info(
                "Training finished. (run_name %s, run_id %s)",
                run_name,
                run_id,
            )
        _cleanup_distributed(distributed)


@app.command()
def main(
    config_path: str = typer.Option(..., help="Path to the config file."),
    output_dir: str = typer.Option(..., help="Path to the output directory."),
    device: str | None = typer.Option(None, help="Device to use for training."),
    project_name: str | None = typer.Option(None, help="Project name for wandb."),
    wandb_id: str | None = typer.Option(None, help="User/entity ID for wandb."),
    exp_name: str | None = typer.Option(None, help="Run/experiment name for wandb."),
    debug: bool = typer.Option(False, help="Run one batch and eval offline."),
) -> None:
    """Entrypoint for Protpardelle-1c training."""
    train(
        config_path,
        output_dir,
        device=device,
        project_name=project_name,
        wandb_id=wandb_id,
        exp_name=exp_name,
        debug=debug,
    )


if __name__ == "__main__":
    app()
