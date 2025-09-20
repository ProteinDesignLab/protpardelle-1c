"""Entrypoint for Protpardelle-1c likelihood computation.

Authors: Alex Chu, Tianyu Lu
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.core import diffusion
from protpardelle.core.models import load_model
from protpardelle.data import dataset
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.env import (
    PROJECT_ROOT_DIR,
    PROTPARDELLE_MODEL_CONFIGS,
    PROTPARDELLE_MODEL_WEIGHTS,
    PROTPARDELLE_OUTPUT_DIR,
)
from protpardelle.utils import get_logger, seed_everything, unsqueeze_trailing_dims

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def get_backbone_mask(atom_mask):
    backbone_mask = torch.zeros_like(atom_mask)
    for atom in ("N", "CA", "C", "O"):
        backbone_mask[..., residue_constants.atom_order[atom]] = 1
    return backbone_mask


def batch_from_pdbs(list_of_pdbs):
    all_feats = [load_feats_from_pdb(pdb)[0] for pdb in list_of_pdbs]
    max_len = max(f["aatype"].shape[0] for f in all_feats)
    dict_of_lists = {"seq_mask": []}
    for feats in all_feats:
        for k, v in feats.items():
            if k in ["atom_mask", "atom_positions", "residue_index", "chain_index"]:
                if k == "atom_positions":
                    v = dataset.apply_random_se3(
                        v, atom_mask=feats["atom_mask"], translation_scale=0
                    )
                padded_feat, seq_mask = dataset.make_fixed_size_1d(v, max_len)
                dict_of_lists.setdefault(k, []).append(padded_feat)
        dict_of_lists["seq_mask"].append(seq_mask)
    return {k: torch.stack(v) for k, v in dict_of_lists.items()}


def forward_ode(
    model,
    batch,
    num_steps=100,
    sigma_min=0.01,
    sigma_max=800,
    verbose=False,
    eps=None,
):
    """Solve the probability flow ODE to get latent encodings and likelihoods.

    Usage: given a model and a list of pdb paths paths
    batch = batch_from_pdbs(paths)
    results = forward_ode(model, batch)
    nats_per_atom = results['npa']
    latents = results['encoded_latent']

    Based on https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py
    See also https://github.com/crowsonkb/k-diffusion/blob/cc49cf6182284e577e896943f8e29c7c9d1a7f2c/k_diffusion/sampling.py#L281
    """
    device = model.device
    sigma_data = model.sigma_data

    seq_mask = batch["seq_mask"].to(device)
    batch_size = seq_mask.shape[0]
    to_batch_size = lambda x: x * torch.ones(batch_size).to(device)
    residue_index = batch["residue_index"].to(device)
    chain_index = batch["chain_index"].to(device)

    if model.task == "backbone":
        atom_mask = get_backbone_mask(batch["atom_mask"]) * batch["atom_mask"]
    else:
        atom_mask = batch["atom_mask"]
    atom_mask = (torch.ones_like(batch["atom_positions"]) * atom_mask.unsqueeze(-1)).to(
        device
    )

    init_coords = batch["atom_positions"].to(device)
    init_coords = init_coords - torch.mean(
        init_coords[..., 1:2, :], dim=-3, keepdim=True
    )
    random_rots = torch.stack(
        [dataset.uniform_rand_rotation(1)[0].to(device) for _ in range(batch_size)]
    )
    init_coords = torch.einsum("bij,blnj->blni", random_rots, init_coords)

    init_coords = init_coords * atom_mask
    batch_data_sizes = atom_mask.sum((1, 2, 3))

    # Noise for skilling-hutchinson
    if eps is None:
        eps = torch.randn_like(init_coords)
    sum_dlogp = to_batch_size(0)

    # Initialize noise schedule/parameters
    noise_schedule = lambda t: diffusion.noise_schedule(
        t, s_min=sigma_min / sigma_data, s_max=sigma_max / sigma_data
    )
    timesteps = torch.linspace(0, 1, num_steps + 1)
    sigma = noise_schedule(timesteps[0])

    # init to sigma_min
    xt = init_coords + torch.randn_like(init_coords) * sigma

    sigma = to_batch_size(sigma)
    xt_traj = []

    def dx_dt_f_theta(xt, sigma, x0=None):
        xt = xt * atom_mask

        # For allatom-nomask models, need to inject fresh Gaussian noise
        # at dummy atom dimensions at the current noise level
        if model.task == "ai-allatom-nomask":
            dummy_fill_mask = 1 - atom_mask
            if x0 is not None:
                dummy_fill_noise = (
                    torch.randn_like(xt) * unsqueeze_trailing_dims(sigma, xt)
                ) + x0[:, :, 1:2, :]
            else:
                dummy_fill_noise = (
                    torch.randn_like(xt) * unsqueeze_trailing_dims(sigma, xt)
                ) + xt[:, :, 1:2, :]
            xt = xt * atom_mask
            xt = xt + dummy_fill_noise * dummy_fill_mask

        x0, _, _, _ = model.forward(
            noisy_coords=xt,
            noise_level=sigma,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            run_mpnn_model=False,
        )
        dx_dt = (xt - x0) / unsqueeze_trailing_dims(sigma, xt)
        dx_dt = dx_dt * atom_mask
        return dx_dt, x0

    # Forward PF ODE
    with torch.no_grad():
        x0 = None
        for t in iter(timesteps[1:]):
            sigma_next = noise_schedule(t)
            sigma_next = to_batch_size(sigma_next)
            step_size = sigma_next - sigma

            # Euler integrator
            with torch.enable_grad():
                xt.requires_grad_(True)
                dx_dt, x0 = dx_dt_f_theta(xt, sigma, x0=x0)
                hutch_proj = (dx_dt * eps * atom_mask).sum()
                grad = torch.autograd.grad(hutch_proj, xt)[0]
            xt.requires_grad_(False)
            dx = dx_dt * unsqueeze_trailing_dims(step_size, dx_dt)
            xt = xt + dx
            dlogp_dt = (grad * eps * atom_mask).sum((1, 2, 3))
            dlogp = dlogp_dt * unsqueeze_trailing_dims(step_size, dlogp_dt)
            sum_dlogp = sum_dlogp + dlogp

            sigma = sigma_next

            # Logging
            xt_scale = sigma_data / unsqueeze_trailing_dims(
                torch.sqrt(sigma_next**2 + sigma_data**2), xt
            )
            scaled_xt = xt * xt_scale
            xt_traj.append(scaled_xt.cpu())

    prior_logp = -1 * batch_data_sizes / 2.0 * np.log(2 * np.pi * sigma_max**2) - (
        xt * xt
    ).sum((1, 2, 3)) / (2 * sigma_max**2)

    logp = prior_logp + sum_dlogp
    nats_per_atom = -logp / batch_data_sizes * 3
    bits_per_dim = -logp / batch_data_sizes / np.log(2)

    results = {
        "prior_logp": prior_logp,
        "prior_logp_per_atom": prior_logp / batch_data_sizes * 3,
        "deltalogp": sum_dlogp,
        "deltalogp_per_atom": sum_dlogp / batch_data_sizes * 3,
        "logp": logp,
        "nats_per_atom": nats_per_atom,
        "bits_per_dim": bits_per_dim,
        "batch_data_sizes": batch_data_sizes,
        "protein_lengths": seq_mask.sum(-1),
        "encoded_latent": xt,
        "seq_mask": seq_mask,
    }
    if verbose:
        for k, v in results.items():
            logger.info("%s: %s", k, v)
    return results


def runner(
    model_name: str = "cc58",
    epoch: str = "416",
    pdb_path: Path = PROJECT_ROOT_DIR / "examples/motifs/nanobody",
    batch_size: int = 32,
    seed: int | None = None,
):
    """Run the likelihood computation for given PDB files using a specified model.

    Args:
        model_name (str): The name of the model to use (e.g., 'cc58', 'cc89').
            This corresponds to a model configuration file.
        epoch (str): The epoch number of the model checkpoint to load.
        pdb_path (Path): The path to a single .pdb file or a directory
            containing multiple .pdb files.
        batch_size (int, optional): The number of samples to process in each batch.
            Defaults to 32.
        seed (int | None, optional): A random seed for reproducibility.
            Defaults to None.

    Examples:
        python -m protpardelle.likelihood --model-name cc58 --epoch 416 --pdb-path ./examples/motifs/nanobody
    """

    if seed is not None:
        seed_everything(seed)

    config_path = PROTPARDELLE_MODEL_CONFIGS / f"{model_name}.yaml"
    checkpoint_path = PROTPARDELLE_MODEL_WEIGHTS / f"{model_name}_epoch{epoch}.pth"

    model = load_model(config_path, checkpoint_path)

    if pdb_path.is_dir():
        pdb_paths = list(pdb_path.glob("*.pdb"))
    else:
        pdb_paths = [pdb_path]
    pdb_stems = [pdb_fp.stem for pdb_fp in pdb_paths]

    num_samples = len(pdb_paths)

    all_results = defaultdict(list)

    batch_sizes = [batch_size] * (num_samples // batch_size)
    if num_samples % batch_size != 0:
        batch_sizes.append(num_samples % batch_size)

    for i, bs in tqdm(enumerate(batch_sizes)):

        si, ei = i * bs, (i + 1) * bs
        if i == len(batch_sizes) - 1:
            si, ei = -bs, num_samples

        batch = batch_from_pdbs(pdb_paths[si:ei])
        results = forward_ode(model, batch)

        for k, v in results.items():
            all_results[k].extend(v)

    save_dir = (
        PROTPARDELLE_OUTPUT_DIR / f"likelihood_{model_name}_{epoch}" / pdb_path.stem
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    latents = all_results.pop("encoded_latent")
    seq_masks = all_results.pop("seq_mask")
    latent_save_dir = save_dir / "latents"
    latent_save_dir.mkdir(exist_ok=True, parents=True)
    for i, latent in enumerate(latents):
        curr_seq_mask = torch.nonzero(seq_masks[i]).flatten()
        torch.save(
            latent[curr_seq_mask], latent_save_dir / f"{pdb_stems[i]}_encoded_latent.pt"
        )

    for k, v in all_results.items():
        all_results[k] = [value.item() for value in v]

    df = pd.DataFrame(all_results)
    df["pdb_path"] = pdb_paths
    df["pdb_stem"] = pdb_stems
    df.to_csv(save_dir / "likelihood_result.csv", float_format="%.4f", index=False)

    logger.info("Likelihood results saved to %s", save_dir / "likelihood_result.csv")
    logger.info("Latents saved to %s", latent_save_dir)


@app.command()
def main(
    model_name: str = typer.Option("cc58", help="Model name, e.g., 'cc58', 'cc89'"),
    epoch: str = typer.Option("416", help="Epoch number of the model checkpoint"),
    pdb_path: Path = typer.Option(
        PROJECT_ROOT_DIR / "examples/motifs/nanobody",
        help="Path to a .pdb file or a directory containing .pdb files",
    ),
    batch_size: int = typer.Option(32, help="Batch size for processing samples"),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility"),
):
    """Compute likelihoods for given PDB files using a specified Protpardelle model."""
    runner(
        model_name=model_name,
        epoch=epoch,
        pdb_path=pdb_path,
        batch_size=batch_size,
        seed=seed,
    )


if __name__ == "__main__":
    app()
