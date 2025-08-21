"""Noise and diffusion utils.

Authors: Alex Chu, Tianyu Lu
"""

import torch
from scipy.stats import norm
from torchtyping import TensorType

from protpardelle import utils


def noise_schedule(
    time,
    function: str = "uniform",
    sigma_data: float = 10.0,
    psigma_mean: float = -0.5,
    psigma_std: float = 1.5,
    s_min: float = 0.001,
    s_max: float = 80,
    rho: float = 7.0,
    time_power: float = 4.0,
    constant_val: float = 0.0,
):
    def sampling_noise(time):
        # high noise = 1; low noise = 0. opposite of Karras et al. schedule
        term1 = s_max ** (1 / rho)
        term2 = (1 - time) * (s_min ** (1 / rho) - s_max ** (1 / rho))
        noise_level = sigma_data * ((term1 + term2) ** rho)
        return noise_level

    if function == "lognormal":
        normal_sample = torch.Tensor(norm.ppf(time.cpu())).to(time)
        noise_level = sigma_data * torch.exp(psigma_mean + psigma_std * normal_sample)
    elif function == "uniform":
        noise_level = sampling_noise(time)
    elif function == "mpnn":
        time = time**time_power
        noise_level = sampling_noise(time)
    elif function == "constant":
        noise_level = torch.ones_like(time) * constant_val
    return noise_level


def noise_coords(
    coords: TensorType["b n a x", float],
    noise_level: TensorType["b", float],
    atom_mask: TensorType["b n a"] = None,
    dummy_fill_mode: str = "zero",
):
    assert atom_mask is not None
    dummy_fill_mask = 1 - atom_mask
    if dummy_fill_mode == "zero":
        dummy_fill_value = 0
    else:
        dummy_fill_value = coords[..., 1:2, :]  # CA
    coords = (
        coords * atom_mask[..., None] + dummy_fill_value * dummy_fill_mask[..., None]
    )

    noise = torch.randn_like(coords) * utils.unsqueeze_trailing_dims(
        noise_level, coords
    )
    noisy_coords = coords + noise
    return noisy_coords
