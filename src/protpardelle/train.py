"""General training script.

Authors: Alex Chu, Richard Shuai
"""

import argparse
import datetime
import os
import random
import shlex
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm.auto import tqdm

import protpardelle.core.diffusion as diffusion
import protpardelle.core.models as models
import protpardelle.core.modules as modules
import protpardelle.utils
from protpardelle.common import residue_constants
from protpardelle.data import atom
from protpardelle.data import dataset as protpardelle_dataset
from protpardelle.utils import unsqueeze_trailing_dims


@record
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=str, default="other", help="wandb project name"
    )
    parser.add_argument("--wandb_id", type=str, default="", help="wandb username")
    parser.add_argument("--exp_name", type=str, default=None, help="wandb exp name")
    parser.add_argument(
        "--config", type=str, default="configs/config.yml", help="experiment config"
    )
    parser.add_argument(
        "--train", default=False, action="store_true", help="dont run in debug mode"
    )
    parser.add_argument(
        "--overfit", type=int, default=-1, help="number of examples to overfit to"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="run one batch and eval offline",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="do not prepend debug to output dirs",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--n_gpu_per_node", type=int, default=1, help="num gpus per node"
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="num nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="rank amongst nodes")
    parser.add_argument(
        "--use_dataparallel",
        default=False,
        action="store_true",
        help="use DataParallel",
    )
    parser.add_argument(
        "--use_ddp",
        default=False,
        action="store_true",
        help="use DistributedDataParallel",
    )
    parser.add_argument(
        "--detect_anomaly", default=False, action="store_true", help="detect nans"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader num workers"
    )
    parser.add_argument(
        "--use_amp",
        default=False,
        action="store_true",
        help="automatic mixed precision",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="output path for trained models"
    )
    opt = parser.parse_args()

    if opt.use_ddp:
        opt.world_size = opt.n_gpu_per_node * opt.n_nodes
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=5400)
        )
        dist.barrier()

    train(opt)

    return


def masked_cross_entropy(logprobs, target, loss_mask):
    # target is onehot
    cel = -(target * logprobs)
    cel = cel * loss_mask[..., None]
    cel = cel.sum((-1, -2)) / loss_mask.sum(-1).clamp(min=1e-6)
    return cel


def masked_mse(x, y, mask, weight=None):
    data_dims = tuple(range(1, len(x.shape)))
    mse = (x - y).pow(2) * mask
    if weight is not None:
        mse = mse * unsqueeze_trailing_dims(weight, mse)
    mse = mse.sum(data_dims) / mask.sum(data_dims).clamp(min=1e-6)
    return mse


class ProtpardelleRunner:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataloader,
        save_dir,
        device,
        scaler=None,
    ):
        self.config = config

        if isinstance(model, nn.DataParallel) or isinstance(
            model, nn.parallel.DistributedDataParallel
        ):
            self.model = model.module
        else:
            self.model = model
        self.forward = model

        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler()
        self.dataset = train_dataset
        self.eval_dataloader = eval_dataloader
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
        time=None,
        is_training=False,
        return_aux=False,
    ):
        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        aatype = inputs["aatype"]
        aatype_oh = F.one_hot(aatype, self.config.data.n_aatype_tokens).float()
        atom_mask = inputs["atom_mask"]
        device = coords.device
        bs = coords.shape[0]

        struct_crop_cond = None
        if self.config.train.crop_conditional:
            assert (
                not self.config.model.compute_loss_on_all_atoms
            ), "Crop conditioning with compute_loss_on_all_atoms not implemented"
            coords, crop_cond_mask, hotspot_mask = (
                protpardelle_dataset.make_crop_cond_mask_and_recenter_coords(
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
        if time is None:
            time = torch.rand(bs).clamp(min=1e-9, max=1 - 1e-9).to(device)
        noise_level = self.model.training_noise_schedule(time)

        noised_coords = diffusion.noise_coords(
            coords,
            noise_level,
            atom_mask=atom_mask,
            dummy_fill_mode=self.config.data.dummy_fill_mode,
        )

        bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
        bb_atom_mask = atom.atom37_mask_from_aatype(bb_seq, seq_mask)

        #! some backbone atoms may be missing -- mask them to zeros!
        bb_atom_mask = torch.logical_and(bb_atom_mask, atom_mask)

        if self.config.model.task == "backbone":
            noised_coords *= bb_atom_mask[..., None]
        elif self.config.model.task == "ai-allatom":
            noised_coords *= atom_mask[..., None]
        elif self.config.model.task == "ai-allatom-hybrid":
            hybrid_mask = torch.ones_like(atom_mask)
            hybrid_mask[time > 0.5] *= bb_atom_mask[time > 0.5]
            noised_coords *= hybrid_mask[..., None]

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

        loss = 0.0
        aux = {}

        # Compute structure loss
        if self.config.model.task in [
            "backbone",
            "allatom",
            "ai-allatom",
            "ai-allatom-nomask",
            "ai-allatom-hybrid",
            "codesign",
        ]:
            if self.config.model.task == "backbone":
                struct_loss_mask = torch.ones_like(coords) * bb_atom_mask[..., None]
            else:
                if self.config.model.compute_loss_on_all_atoms:
                    # Compute loss on all 37 atoms
                    struct_loss_mask = torch.ones_like(
                        coords
                    ) * protpardelle.utils.unsqueeze_trailing_dims(seq_mask, coords)
                elif self.config.model.task == "ai-allatom-hybrid":
                    struct_loss_mask = torch.ones_like(coords) * hybrid_mask[..., None]
                else:
                    struct_loss_mask = torch.ones_like(coords) * atom_mask[..., None]
            loss_weight = (noise_level**2 + self.sigma_data**2) / (
                (noise_level * self.sigma_data) ** 2
            )
            struct_loss = masked_mse(
                coords, denoised_coords, struct_loss_mask, loss_weight
            )
            loss += struct_loss
            aux["struct_loss"] = struct_loss.mean().detach().cpu().item()

        # Compute mpnn loss
        if self.config.model.task in ["seqdes", "codesign"]:
            alpha = self.config.model.mpnn_model.label_smoothing
            target_oh = (1 - alpha) * aatype_oh + alpha / self.model.n_tokens
            seq_loss_mask = seq_mask
            mpnn_loss = masked_cross_entropy(
                pred_seq_logprobs, target_oh, seq_loss_mask
            )
            loss += mpnn_loss
            aux["mpnn_loss"] = mpnn_loss.mean().detach().cpu().item()

        aux["train_loss"] = loss.mean().detach().cpu().item()
        if return_aux:
            return loss.mean(), aux
        return loss.mean()

    def train_step(self, inputs):
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
                log_dict["grad_norm"] = grad_norm

                try:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip_val
                    )
                except Exception:
                    pass

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
            log_dict["grad_norm"] = grad_norm

            try:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip_val
                )
            except Exception:
                pass
            self.optimizer.step()
            self.scheduler.step()

        # Add train prefix to all keys
        keys = list(log_dict.keys())
        for k in keys:
            log_dict[f"train/{k}"] = log_dict.pop(k)

        return log_dict


def train(opt):
    config, config_dict = protpardelle.utils.load_config(opt.config, return_dict=True)
    wandb_dir = str(Path(opt.out_dir, opt.project))
    Path(wandb_dir, "wandb").mkdir(parents=True, exist_ok=True)  # Create wandb dir

    # Set wandb cache directory
    wandb_cache_dir = str(Path(opt.out_dir, opt.project, "cache", "wandb"))
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

    # TODO: Fix DDP
    if opt.use_ddp:
        raise NotImplementedError(
            "DDP not implemented yet (specifically since we're using the StochasticMixedSampler)"
        )

    # Set up devices
    if opt.use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{rank}"
        assert dist.is_available() and dist.is_initialized(), (
            dist.is_available(),
            dist.is_initialized(),
        )
    elif not opt.no_cuda and torch.cuda.is_available():
        torch.cuda.init()
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        device = f"cuda:{opt.gpu_id}"
    else:
        device = "cpu"

    # Set seeds
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.deterministic = True
    # nonrandom selection of CUDNN convolution, maybe slower
    torch.backends.cudnn.benchmark = False

    # Set up datasets
    def get_dataloader(mode):
        # Load in datasets, using torch concatenation to combine datasets if there are multiple specified
        # assumes the first dataset is the main dataset for computing epochs, sample from the other datasets with replacement
        if mode == "train":
            bs = config.train.batch_size
            train_datasets = [
                protpardelle_dataset.PDBDataset(
                    pdb_path=config.data.pdb_paths[di],
                    fixed_size=config.data.fixed_size,
                    mode=mode,
                    overfit=opt.overfit,
                    short_epoch=not opt.train,
                    se3_data_augment=config.data.se3_data_augment,
                    translation_scale=config.data.translation_scale,
                    chain_residx_gap=config.data.chain_residx_gap,
                    dummy_fill_mode=config.data.dummy_fill_mode,
                    subset=config.data.subset[di],
                )
                for di in range(len(config.data.pdb_paths))
            ]
            dataset = torch.utils.data.ConcatDataset(train_datasets)
            train_sampler = protpardelle_dataset.StochasticMixedSampler(
                train_datasets, config.data.mixing_ratios, batch_size=bs
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                num_workers=opt.num_workers,
                pin_memory="cuda" in device,
                shuffle=False,  # the sampler takes care of shuffling
                sampler=train_sampler,
                drop_last=True,
                persistent_workers=False if opt.debug else True,
            )
        elif mode == "eval":
            # for evaluation, we use the primary dataset (the first one)
            bs = 1
            dataset = protpardelle_dataset.PDBDataset(
                pdb_path=config.data.pdb_paths[0],
                fixed_size=config.data.fixed_size,
                mode=mode,
                overfit=opt.overfit,
                short_epoch=not opt.train,
                se3_data_augment=config.data.se3_data_augment,
                translation_scale=config.data.translation_scale,
                chain_residx_gap=config.data.chain_residx_gap,
                dummy_fill_mode=config.data.dummy_fill_mode,
                subset=config.data.subset[0],
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                num_workers=opt.num_workers,
                pin_memory="cuda" in device,
                shuffle=False,
                persistent_workers=False if opt.debug else True,
            )
        return dataset, dataloader

    dataset, dataloader = get_dataloader("train")
    eval_dataset, eval_dataloader = get_dataloader("eval")

    # Calculate sigma_data
    if config.data.auto_calc_sigma_data:
        print(
            f"===== Automatically computing sigma_data for {config.data.n_examples_for_sigma_data} examples ====="
        )
        sigma_data = protpardelle_dataset.calc_sigma_data(
            dataset, config, opt.num_workers
        )

        # Update config and config_dict with estimated sigma_data
        config.data.sigma_data = sigma_data
        config_dict["data"]["sigma_data"] = sigma_data

    # Init wandb and logging for process 0
    log_dir = ""
    if not opt.use_ddp or rank == 0:
        if opt.train:
            wandb.init(
                mode="disabled" if opt.debug else "online",
                project=opt.project,
                entity=opt.wandb_id,
                name=opt.exp_name,
                job_type="train",
                config=config_dict,
                dir=wandb_dir,
            )
        else:
            wandb.init(
                mode="disabled" if opt.debug else "online",
                project=opt.project,
                entity=opt.wandb_id,
                name=opt.exp_name,
                job_type="debug",
                config=config_dict,
                dir=wandb_dir,
            )
        print(
            f'Beginning; run name "{wandb.run.name}", run id "{wandb.run.id}", device {device}'
        )
        print(opt)

    # Set up logging
    if opt.train:
        log_dir = Path(opt.out_dir, opt.project, wandb.run.name)
    else:
        log_dir = Path(opt.out_dir, opt.project, f"debug_{wandb.run.name}")

    Path(log_dir, "results").mkdir(parents=True, exist_ok=True)
    # Path(log_dir, "checkpoints").mkdir(parents=True, exist_ok=config.train.ckpt_path != "") #! CHANGED
    Path(log_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(log_dir, "config.yml"), "w") as f:
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

    if opt.use_dataparallel:
        model = nn.DataParallel(model)
    if opt.use_ddp:
        model.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    model.train()
    model.to(device)

    runner = ProtpardelleRunner(
        config,
        model,
        dataset,
        eval_dataloader,
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

    if not opt.use_ddp or rank == 0:
        start_time = time.time()
    with (
        torch.autograd.set_detect_anomaly(True) if opt.detect_anomaly else nullcontext()
    ):
        for epoch in range(start_epoch + 1, config.train.max_epochs + 1):
            if opt.use_ddp:
                dist.barrier()
                dataloader.sampler.set_epoch(epoch)
                eval_dataloader.sampler.set_epoch(epoch)

            for inputs in tqdm(
                dataloader, desc=f"epoch {epoch}/{config.train.max_epochs}"
            ):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["step"] = total_steps
                log_dict = runner.train_step(inputs)
                log_dict["learning_rate"] = runner.scheduler.get_last_lr()[0]
                if not opt.use_ddp or rank == 0:
                    wandb.log(log_dict, step=total_steps)
                    wandb.log({"epoch": epoch}, step=total_steps)
                total_steps += 1
                if opt.debug:
                    break

            with torch.no_grad():  # per epoch
                # Run eval and save checkpoint
                if opt.use_ddp:
                    dist.barrier()
                if (
                    epoch % config.train.checkpoint_freq == 0
                    or epoch in config.train.checkpoints
                ):
                    if not opt.use_ddp or rank == 0:
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

    if not opt.use_ddp or rank == 0:
        wandb.finish()
        subprocess.run(shlex.split(f"cp -r {wandb.run.dir} {log_dir}"))
        print(
            f'Training finished. (run name "{wandb.run.name}", run id "{wandb.run.id}")'
        )
    if opt.use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
