"""Entrypoint for Protpardelle-1c training.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li, Tianyu Lu
"""

import argparse
import logging
import random
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import wandb
import yaml
from torch.amp import GradScaler, autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.types import Device
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.core import diffusion, modules
from protpardelle.core.models import Protpardelle
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.dataset import (
    PDBDataset,
    StochasticMixedSampler,
    calc_sigma_data,
    make_crop_cond_mask_and_recenter_coords,
)
from protpardelle.utils import (
    StrPath,
    get_default_device,
    load_config,
    namespace_to_dict,
    norm_path,
    seed_everything,
    unsqueeze_trailing_dims,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def masked_cross_entropy_loss(
    logprobs: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, tol: float = 1e-7
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
    cel = cel * loss_mask.unsqueeze(-1)
    cel = cel.sum((-1, -2)) / loss_mask.sum(-1).clamp(min=tol)

    return cel


def masked_mse_loss(
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
    mse = (x - y).square() * mask
    if weights is not None:
        mse = mse * unsqueeze_trailing_dims(weights, mse)
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=tol)

    return mse


class ProtpardelleTrainer:
    """Trainer for the Protpardelle model."""

    def __init__(
        self,
        config: argparse.Namespace,
        device: Device,
    ) -> None:
        """Initialize the ProtpardelleTrainer.

        Args:
            config (argparse.Namespace): The training configuration.
            device (Device): The device to train on.
        """

        self.config = config
        if device is None:
            device = get_default_device()

        # Initialize model
        model = Protpardelle(config, device)
        if torch.cuda.is_available():
            logger.info("Device count: %d", torch.cuda.device_count())
            logger.info("Current device: %d", torch.cuda.current_device())
            if torch.cuda.device_count() > 1:
                # TODO: Implement DDP
                model = nn.DataParallel(model)  # type: ignore
                logger.info("Using DataParallel")
        model.to(device)
        model.train()
        self.model = model

        # Initialize optimizer, scheduler, and scaler
        self.optimizer, self.scheduler = self._load_optimizer_and_scheduler()
        self.scaler = GradScaler(device=device, enabled=config.train.use_amp)  # type: ignore

    @property
    def module(self) -> Protpardelle:
        """Get the underlying Protpardelle model on the fly."""
        return (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
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
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "total_steps": total_steps,
            "pytorch_version": torch.__version__,
        }
        checkpoint["rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        checkpoint_dir = norm_path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "epoch": epoch,
                "total_steps": total_steps,
                "pytorch_version": torch.__version__,
            },
            checkpoint_dir / f"epoch{epoch}_training_state.pth",
        )

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

        torch.set_rng_state(checkpoint["rng"]["torch"])
        if torch.cuda.is_available():
            if checkpoint["rng"]["cuda"] is None:
                raise ValueError(
                    "Checkpoint was trained with CUDA but current device is CPU"
                )
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        random.setstate(checkpoint["rng"]["python"])

        return checkpoint["epoch"], checkpoint["total_steps"]

    def log_training_info(self) -> None:
        """Log training information."""
        logger.info(
            "Total params: %d", sum(p.numel() for p in self.module.parameters())
        )
        logger.info(
            "Trainable params: %d",
            sum(p.numel() for p in self.module.parameters() if p.requires_grad),
        )

    def get_dataloader(
        self, overfit: int = 0, num_workers: int = 0, debug: bool = False
    ) -> DataLoader:
        """Get the training dataloader.

        Args:
            overfit (int, optional): If > 0, will overfit to this many examples; 0 means no overfitting.
                Defaults to 0.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            debug (bool, optional): If True, will use a smaller dataset for debugging. Defaults to False.

        Returns:
            DataLoader: The training dataloader.
        """

        # Initialize and combine training datasets. The StochasticMixedSampler will handle
        # sampling from the combined datasets according to specified mixing ratios.
        train_datasets = [
            PDBDataset(
                pdb_path=self.config.data.pdb_paths[di],
                fixed_size=self.config.data.fixed_size,
                mode="train",
                overfit=overfit,
                short_epoch=debug,
                se3_data_augment=self.config.data.se3_data_augment,
                translation_scale=self.config.data.translation_scale,
                chain_residx_gap=self.config.data.chain_residx_gap,
                dummy_fill_mode=self.config.data.dummy_fill_mode,
                subset=self.config.data.subset[di],
            )
            for di, _ in enumerate(self.config.data.pdb_paths)
        ]
        dataset: ConcatDataset[PDBDataset] = ConcatDataset(train_datasets)

        sampler = StochasticMixedSampler(
            train_datasets,
            self.config.data.mixing_ratios,
            batch_size=self.config.train.batch_size,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=False,  # the sampler takes care of shuffling
            sampler=sampler,
            drop_last=True,
            persistent_workers=not debug,
        )

        if self.config.data.auto_calc_sigma_data:
            self.compute_sigma_data(dataset)

        return dataloader

    def compute_sigma_data(self, dataset: Dataset, num_workers: int = 0) -> float:
        """Compute the sigma data for the given dataset.

        Args:
            dataset (Dataset): The dataset to compute sigma data for.
            num_workers (int, optional): The number of workers to use for data loading. Defaults to 0.

        Returns:
            float: The computed sigma data.
        """

        logger.info(
            "Automatically computing sigma_data for %d examples",
            self.config.data.n_examples_for_sigma_data,
        )
        sigma_data = calc_sigma_data(dataset, self.config, num_workers=num_workers)

        # Overwrite config with estimated sigma_data
        self.config.data.sigma_data = sigma_data

        return sigma_data

    def compute_loss(
        self, input_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the loss for a given input batch.

        Args:
            input_dict (dict[str, torch.Tensor]): Input tensors for the model.

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
            atom_coords, crop_cond_mask, hotspot_mask = (
                make_crop_cond_mask_and_recenter_coords(
                    atom_mask,
                    atom_coords,
                    aatype=aatype,
                    chain_index=chain_index,
                    **vars(self.config.train.crop_cond),
                )
            )
            struct_crop_cond = atom_coords * crop_cond_mask.unsqueeze(-1)
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
        timestep = torch.rand(batch_size, device=self.device)
        noise_level = self.module.training_noise_schedule(timestep)
        noised_coords = diffusion.noise_coords(
            atom_coords,
            noise_level,
            atom_mask,
            dummy_fill_mode=self.config.data.dummy_fill_mode,
        )

        bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
        bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)

        # Some backbone atoms may be missing; mask them to zeros
        bb_atom_mask = torch.logical_and(
            bb_atom_mask, atom_mask
        )  # both masks are float; cannot use &
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

        if np.random.uniform() < self.config.train.self_cond_train_prob:
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

        loss = torch.tensor(0.0, device=self.device, dtype=torch.float)
        log_dict: dict[str, float] = {}

        # Compute structure loss
        if self.config.model.task in [
            "backbone",
            "allatom",
            "ai-allatom",
            "ai-allatom-nomask",
            "codesign",
        ]:
            if self.config.model.task == "backbone":
                struct_loss_mask = torch.ones_like(atom_coords) * bb_atom_mask.unsqueeze(-1)
            else:
                if self.config.model.compute_loss_on_all_atoms:
                    # Compute loss on all 37 atoms
                    struct_loss_mask = torch.ones_like(
                        atom_coords
                    ) * unsqueeze_trailing_dims(seq_mask, atom_coords)
                else:
                    struct_loss_mask = torch.ones_like(atom_coords) * atom_mask.unsqueeze(-1)
            loss_weight = (noise_level**2 + self.module.sigma_data**2) / (
                (noise_level * self.module.sigma_data) ** 2
            )
            struct_loss = masked_mse_loss(
                atom_coords, denoised_coords, struct_loss_mask, loss_weight
            ).mean()
            loss = loss + struct_loss
            log_dict["struct_loss"] = struct_loss.detach().cpu().item()

        # Compute mpnn loss
        if self.config.model.task in ["seqdes", "codesign"]:
            alpha = self.config.model.mpnn_model.label_smoothing
            aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
            target_oh = (1 - alpha) * aatype_oh + alpha / self.module.n_tokens
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

        with autocast("cuda") if self.config.train.use_amp else nullcontext():
            loss, log_dict = self.compute_loss(input_dict)
            self.scaler.scale(loss).backward()

            # Compute the gradient norm and add it to the log_dict
            grad_norm = nn.utils.clip_grad_norm_(self.module.parameters(), float("inf"))
            log_dict["grad_norm"] = grad_norm.item()

            try:
                nn.utils.clip_grad_norm_(
                    self.module.parameters(),
                    self.config.train.grad_clip_val,
                )
            except RuntimeError as e:
                logger.warning("Failed to clip gradients: %s", e)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        # Add train prefix to all keys for wandb logging
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}

        return log_dict


def _initialize_training_parameters(trainer: ProtpardelleTrainer) -> tuple[int, int]:
    """Initialize training parameters.

    Args:
        trainer (ProtpardelleTrainer): The protpardelle trainer instance.

    Returns:
        tuple[int, int]: The starting epoch and total steps.
    """

    start_epoch = 0
    total_steps = 0

    # Set seeds if no rng provided
    seed = trainer.config.train.seed
    if seed is not None:
        seed_everything(
            seed, freeze_cuda=True
        )  # use deterministic pytorch for training

    return start_epoch, total_steps


def _load_checkpoint_or_not(trainer: ProtpardelleTrainer) -> tuple[int, int]:
    """Load checkpoint if it exists, otherwise initialize training parameters.

    Args:
        trainer (ProtpardelleTrainer): The protpardelle trainer instance.

    Returns:
        tuple[int, int]: The starting epoch and total steps.
    """

    checkpoint_path = trainer.config.train.ckpt_path
    if checkpoint_path is None:
        return _initialize_training_parameters(trainer)

    checkpoint_path = norm_path(checkpoint_path)
    try:
        start_epoch, total_steps = trainer.load_checkpoint(checkpoint_path)
        logger.info(
            "Resumed from checkpoint: %s (epoch %d, total_steps %d)",
            checkpoint_path,
            start_epoch,
            total_steps,
        )
        return start_epoch, total_steps
    except FileNotFoundError:
        logger.warning(
            "Checkpoint file not found: %s; starting from scratch", checkpoint_path
        )

    return _initialize_training_parameters(trainer)


@record
def train(
    config_path: StrPath,
    output_dir: StrPath,
    device: Device = None,
    project_name: str | None = None,
    wandb_id: str | None = None,
    exp_name: str | None = None,
    overfit: int = 0,
    num_workers: int = 0,
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
        overfit (int, optional): Overfit on this many examples for debugging. Defaults to 0.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Raises:
        RuntimeError: If wandb initialization fails.
    """

    config = load_config(config_path)

    if debug:
        logger.debug("Debug mode is enabled")
        num_workers = 0  # override num_workers to 0 in debug mode

    trainer = ProtpardelleTrainer(config, device)

    start_epoch, total_steps = _load_checkpoint_or_not(trainer)

    # Set output directories for wandb
    output_dir = norm_path(output_dir)

    # Create dataloader
    dataloader = trainer.get_dataloader(
        overfit=overfit, num_workers=num_workers, debug=debug
    )

    # Init wandb and logging for process 0
    wandb.init(
        mode="disabled" if debug else "online",
        project=project_name,
        entity=wandb_id,
        name=exp_name,
        job_type="debug" if debug else "train",
        config=trainer.config,  # type: ignore
        dir=output_dir,
    )
    if wandb.run is None:
        raise RuntimeError("Failed to initialize wandb run")
    run_name = wandb.run.name
    run_dir = wandb.run.dir
    run_id = wandb.run.id

    # Assert variables cannot be None if wandb runs properly
    assert run_name is not None
    assert run_dir is not None
    assert run_id is not None

    logger.info(
        "Beginning: run_name=%s, run_id=%s, device=%s",
        run_name,
        run_id,
        trainer.device,
    )
    logger.info(
        "Training configuration: %s, %s, %s", config_path, output_dir, project_name
    )

    log_dir = output_dir / (f"{run_name}_debug" if debug else run_name)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(namespace_to_dict(trainer.config), f)

    trainer.log_training_info()
    with torch.autograd.set_detect_anomaly(True) if debug else nullcontext():
        for epoch in range(start_epoch + 1, trainer.config.train.max_epochs + 1):
            for input_dict in tqdm(
                dataloader, desc=f"epoch {epoch}/{trainer.config.train.max_epochs}"
            ):
                input_dict = {k: v.to(trainer.device) for k, v in input_dict.items()}
                input_dict["step"] = total_steps
                log_dict = trainer.train_step(input_dict)
                log_dict["learning_rate"] = trainer.scheduler.get_last_lr()[0]
                log_dict["epoch"] = epoch
                wandb.log(log_dict, step=total_steps)
                total_steps += 1

            with torch.no_grad():  # per epoch
                # Save checkpoint
                if (
                    epoch % trainer.config.train.checkpoint_freq == 0
                    or epoch in trainer.config.train.checkpoints
                ):
                    trainer.module.eval()
                    trainer.save_checkpoint(epoch, total_steps, checkpoint_dir)
                    trainer.module.train()

    wandb.finish()
    subprocess.run(["cp", "-r", str(run_dir), str(log_dir)], check=False)
    logger.info(
        "Training finished. (run name '%s', run id '%s')",
        run_name,
        run_id,
    )


@app.command()
def main(
    config_path: str = typer.Option(..., help="Path to the config file."),
    output_dir: str = typer.Option(..., help="Path to the output directory."),
    device: str | None = typer.Option(None, help="Device to use for training."),
    project_name: str | None = typer.Option(None, help="Project name for wandb."),
    wandb_id: str | None = typer.Option(None, help="User/entity ID for wandb."),
    exp_name: str | None = typer.Option(None, help="Run/experiment name for wandb."),
    overfit: int = typer.Option(
        0, min=0, help="Number of examples to overfit to (0 disables)."
    ),
    num_workers: int = typer.Option(0, min=0, help="DataLoader num_workers."),
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
        overfit=overfit,
        num_workers=num_workers,
        debug=debug,
    )


if __name__ == "__main__":
    app()
