"""Compute model likelihood.

Authors: Alex Chu, Tianyu Lu
"""

import numpy as np
import torch
import typer

from protpardelle import utils
from protpardelle.common import residue_constants
from protpardelle.data import dataset
from protpardelle.data.pdb_io import load_feats_from_pdb


def get_backbone_mask(atom_mask):
    backbone_mask = torch.zeros_like(atom_mask)
    for atom in ("N", "CA", "C", "O"):
        backbone_mask[..., residue_constants.atom_order[atom]] = 1
    return backbone_mask


def batch_from_pdbs(list_of_pdbs):
    all_feats = []
    for pdb in list_of_pdbs:
        all_feats.append(load_feats_from_pdb(pdb)[0])
    max_len = max([f["aatype"].shape[0] for f in all_feats])
    dict_of_lists = {"seq_mask": []}
    for feats in all_feats:
        for k, v in feats.items():
            if k in ["atom_mask", "atom_positions", "residue_index"]:
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
    n_steps=100,
    sigma_min=0.01,
    sigma_max=800,
    tqdm_pbar=None,
    seed=0,
    verbose=False,
    eps=None,
):
    """Solve the probability flow ODE to get latent encodings and likelihoods.

    Usage: given a backbone model `model` and a list of pdb paths `paths`
    batch = batch_from_pdbs(paths)
    results = forward_ode(model, batch)
    nats_per_atom = results['npa']
    latents = results['encoded_latent']

    Based on https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py
    See also https://github.com/crowsonkb/k-diffusion/blob/cc49cf6182284e577e896943f8e29c7c9d1a7f2c/k_diffusion/sampling.py#L281
    """
    assert model.task == "backbone"
    device = model.device
    sigma_data = model.sigma_data
    torch.manual_seed(seed)

    seq_mask = batch["seq_mask"].to(device)
    to_batch_size = lambda x: x * torch.ones(seq_mask.shape[0]).to(device)
    residue_index = batch["residue_index"].to(device)
    backbone_mask = get_backbone_mask(batch["atom_mask"]) * batch["atom_mask"]
    backbone_mask = torch.ones_like(batch["atom_positions"]) * backbone_mask[..., None]
    init_bb_coords = (batch["atom_positions"] * backbone_mask).to(device)
    backbone_mask = backbone_mask.to(device)
    batch_data_sizes = backbone_mask.sum((1, 2, 3))

    # Noise for skilling-hutchinson
    if eps is None:
        eps = torch.randn_like(init_bb_coords)
    sum_dlogp = to_batch_size(0)

    # Initialize noise schedule/parameters
    noise_schedule = lambda t: diffusion.noise_schedule(
        t, s_min=sigma_min / sigma_data, s_max=sigma_max / sigma_data
    )
    timesteps = torch.linspace(0, 1, n_steps + 1)
    sigma = noise_schedule(timesteps[0])

    # init to sigma_min
    xt = init_bb_coords + torch.randn_like(init_bb_coords) * sigma

    sigma = to_batch_size(sigma)
    if tqdm_pbar is None:
        tqdm_pbar = lambda x: x
    xt_traj, x0_traj = [], []

    def dx_dt_f_theta(xt, sigma, sigma_next):
        xt = xt * backbone_mask
        x0, _, _, _ = model.forward(
            noisy_coords=xt,
            noise_level=sigma,
            seq_mask=seq_mask,
            residue_index=residue_index,
            run_mpnn_model=False,
        )
        dx_dt = (xt - x0) / utils.unsqueeze_trailing_dims(sigma, xt)
        dx_dt = dx_dt * backbone_mask
        return dx_dt

    # Forward PF ODE
    with torch.no_grad():
        for i, t in tqdm_pbar(enumerate(iter(timesteps[1:]))):
            sigma_next = noise_schedule(t)
            sigma_next = to_batch_size(sigma_next)
            step_size = sigma_next - sigma

            # Euler integrator
            with torch.enable_grad():
                xt.requires_grad_(True)
                dx_dt = dx_dt_f_theta(xt, sigma, sigma_next)
                hutch_proj = (dx_dt * eps * backbone_mask).sum()
                grad = torch.autograd.grad(hutch_proj, xt)[0]
            xt.requires_grad_(False)
            dx = dx_dt * utils.unsqueeze_trailing_dims(step_size, dx_dt)
            xt = xt + dx
            div = dlogp_dt = (grad * eps * backbone_mask).sum((1, 2, 3))
            dlogp = dlogp_dt * utils.unsqueeze_trailing_dims(step_size, dlogp_dt)
            sum_dlogp = sum_dlogp + dlogp

            sigma = sigma_next

            # Logging
            xt_scale = sigma_data / utils.unsqueeze_trailing_dims(
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
        "npa": nats_per_atom,
        "bpd": bits_per_dim,
        "batch_data_sizes": batch_data_sizes,
        "protein_lengths": seq_mask.sum(-1),
        "encoded_latent": xt,
    }
    if verbose:
        for k, v in results.items():
            print(k, v)
    return results


def runner():
    pass


if __name__ == "__main__":
    typer.run(runner)
