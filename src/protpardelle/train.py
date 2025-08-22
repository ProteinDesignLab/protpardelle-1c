"""General training script.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li
"""

import shlex
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import wandb
import yaml
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

import protpardelle.core.diffusion as diffusion
import protpardelle.core.models as models
import protpardelle.core.modules as modules
from protpardelle.common import residue_constants
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.dataset import (
    PDBDataset,
    StochasticMixedSampler,
    calc_sigma_data,
    make_crop_cond_mask_and_recenter_coords,
)
from protpardelle.utils import (
    dict_to_namespace,
    seed_everything,
    unsqueeze_trailing_dims,
)


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
    ):
        self.config = config

        if isinstance(model, nn.DataParallel):
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

    def compute_loss(
        self,
        inputs,
        timestep=None,
        is_training=False,
        return_aux=False,
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        aatype = inputs["aatype"]
        aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
        atom_mask = inputs["atom_mask"]
        device = coords.device
        bs = coords.shape[0]

        # Initialize variables that may not be set in all branches
        struct_crop_cond = None
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
                    **vars(self.config.train.crop_cond),
                )
            )
            struct_crop_cond = coords * crop_cond_mask[..., None]

        sse_cond, adj_cond = None, None
        if (
            "conditioning_style" in self.config.model
            and "ssadj" in self.config.model.conditioning_style
        ):
            sse_cond = inputs["sse"]
            adj_cond = inputs["adj"]

        # Noise data
        # Sample time
        if timestep is None:
            timestep = torch.rand(bs, device=device)
        noise_level = self.model.training_noise_schedule(timestep)

        noised_coords = diffusion.noise_coords(
            coords,
            noise_level,
            atom_mask=atom_mask,
            dummy_fill_mode=self.config.data.dummy_fill_mode,
        )

        bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
        bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)

        # some backbone atoms may be missing -- mask them to zeros!
        bb_atom_mask = torch.logical_and(bb_atom_mask, atom_mask)

        if self.config.model.task == "backbone":
            noised_coords = noised_coords * bb_atom_mask[..., None]
        elif self.config.model.task == "ai-allatom":
            noised_coords = noised_coords * atom_mask[..., None]

        # Forward pass
        model_inputs = {
            "noisy_coords": noised_coords,
            "noise_level": noise_level,
            "seq_mask": seq_mask,
            "residue_index": inputs["residue_index"],
            "chain_index": inputs["chain_index"],
            "hotspot_mask": (
                hotspot_mask
                if "hotspot" in self.config.model.conditioning_style
                else None
            ),
            "struct_crop_cond": struct_crop_cond,
            "sse_cond": sse_cond,
            "adj_cond": adj_cond,
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
            denoised_coords, pred_seq_logprobs, _, _ = forward_fn(
                **model_inputs,
                struct_self_cond=struct_self_cond,
                seq_self_cond=seq_self_cond,
            )

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
            loss_weight = (noise_level**2 + self.sigma_data**2) / (
                (noise_level * self.sigma_data) ** 2
            )
            struct_loss = masked_mse(
                coords, denoised_coords, struct_loss_mask, loss_weight
            )
            loss = loss + struct_loss
            aux["struct_loss"] = struct_loss.mean().detach().cpu().item()

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
    num_workers: int = 0,
):

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    if debug:
        print("Debug mode is enabled")
        num_workers = 0  # Override num_workers to 0 in debug mode

    # Set seeds
    seed = config.train.seed
    if seed is not None:
        seed_everything(seed)
    # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.deterministic = True
    # nonrandom selection of CUDNN convolution, maybe slower
    torch.backends.cudnn.benchmark = False

    wandb_dir = str(Path(out_dir, project))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)  # Create wandb dir

    # Set up devices
    if torch.cuda.is_available():
        torch.cuda.init()
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    # Load in datasets, using torch concatenation to combine datasets if there are multiple specified
    # assumes the first dataset is the main dataset for computing epochs, sample from the other datasets with replacement
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
        )
        for di in range(len(config.data.pdb_paths))
    ]
    dataset = ConcatDataset(train_datasets)

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
        mode="disabled" if debug else "online",
        project=project,
        entity=wandb_id,
        name=exp_name,
        job_type="debug" if debug else "train",
        config=config_dict,
        dir=wandb_dir,
    )
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

    log_dir = Path(out_dir, project, f"{run_name}_debug" if debug else run_name)

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

    if use_dataparallel:
        model = nn.DataParallel(model)

    model.train()
    model.to(device)

    runner = ProtpardelleRunner(
        config,
        model,
        dataset,
        log_dir,
        device,
    )
    if (
        config.train.ckpt_path != ""
        and "ckpt_optim" in config.train
        and config.train.ckpt_optim
    ):
        for _ in range(total_steps):
            runner.scheduler.step()
        runner.optimizer.load_state_dict(training_state["optim_state_dict"])
    runner.train_init()

    with torch.autograd.set_detect_anomaly(True) if debug else nullcontext():
        for epoch in range(start_epoch + 1, config.train.max_epochs + 1):
            for inputs in tqdm(
                dataloader, desc=f"epoch {epoch}/{config.train.max_epochs}"
            ):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["step"] = total_steps
                log_dict = runner.train_step(inputs)
                log_dict["learning_rate"] = runner.scheduler.get_last_lr()[0]

                wandb.log(log_dict, step=total_steps)
                wandb.log({"epoch": epoch}, step=total_steps)
                total_steps += 1

            with torch.no_grad():  # per epoch
                # Save checkpoint
                if (
                    epoch % config.train.checkpoint_freq == 0
                    or epoch in config.train.checkpoints
                ):
                    runner.model.eval()
                    torch.save(
                        runner.model,
                        f"{log_dir}/checkpoints/epoch{epoch}_model.pth",
                    )
                    torch.save(
                        {
                            "model_state_dict": runner.model.state_dict(),
                            "optim_state_dict": runner.optimizer.state_dict(),
                            "epoch": epoch,
                            "total_steps": total_steps,
                        },
                        f"{log_dir}/checkpoints/epoch{epoch}_training_state.pth",
                    )

                    runner.model.train()

    wandb.finish()
    if wandb.run and wandb.run.dir:
        subprocess.run(shlex.split(f"cp -r {wandb.run.dir} {log_dir}"), check=False)
    if wandb.run:
        print(
            f'Training finished. (run name "{wandb.run.name}", run id "{wandb.run.id}")'
        )
    else:
        print("Training finished.")


def main(
    project: str = typer.Option("other", help="wandb project name"),
    wandb_id: str = typer.Option("", help="wandb username"),
    exp_name: str = typer.Option(None, help="wandb exp name"),
    config: str = typer.Option("configs/config.yml", help="experiment config"),
    overfit: int = typer.Option(-1, help="number of examples to overfit to"),
    debug: bool = typer.Option(False, help="run one batch and eval offline"),
    gpu_id: int = typer.Option(0, help="which GPU to use"),
    use_dataparallel: bool = typer.Option(False, help="use DataParallel"),
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
        num_workers=num_workers,
    )


if __name__ == "__main__":
    typer.run(main)
