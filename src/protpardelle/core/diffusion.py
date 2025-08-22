"""Noise and diffusion utils.

Authors: Alex Chu, Tianyu Lu, Zhaoyang Li
"""

from typing import Literal

import torch
from scipy.stats import norm

from protpardelle.utils import unsqueeze_trailing_dims


def compute_sampling_noise(
    timestep: torch.Tensor,
    sigma_data: float = 10.0,
    s_min: float = 0.001,
    s_max: float = 80,
    rho: float = 7.0,
) -> torch.Tensor:
    """Computes the sampling noise level for a given timestep.

    We set high noise = 1; low noise = 0. opposite of Karras et al. schedule

    Args:
        timestep (torch.Tensor): The current timestep.
        sigma_data (float, optional): The data noise level. Defaults to 10.0.
        s_min (float, optional): The minimum noise level. Defaults to 0.001.
        s_max (float, optional): The maximum noise level. Defaults to 80.
        rho (float, optional): The noise schedule exponent. Defaults to 7.0.

    Returns:
        torch.Tensor: The computed noise level.
    """

    term1 = s_max ** (1 / rho)
    term2 = (1 - timestep) * (s_min ** (1 / rho) - s_max ** (1 / rho))
    noise_level = sigma_data * ((term1 + term2) ** rho)

    return noise_level


def noise_schedule(
    timestep: torch.Tensor,
    function: Literal["uniform", "lognormal", "mpnn", "constant"] = "uniform",
    sigma_data: float = 10.0,
    psigma_mean: float = -0.5,
    psigma_std: float = 1.5,
    s_min: float = 0.001,
    s_max: float = 80,
    rho: float = 7.0,
    time_power: float = 4.0,
    constant_val: float = 0.0,
) -> torch.Tensor:
    """Computes the noise schedule for a given timestep.

    Args:
        timestep (torch.Tensor): The current timestep.
        function (Literal["uniform", "lognormal", "mpnn", "constant"], optional): The noise schedule function to use. Defaults to "uniform".
        sigma_data (float, optional): The data noise level. Defaults to 10.0.
        psigma_mean (float, optional): The mean of the log-normal distribution. Defaults to -0.5.
        psigma_std (float, optional): The standard deviation of the log-normal distribution. Defaults to 1.5.
        s_min (float, optional): The minimum noise level. Defaults to 0.001.
        s_max (float, optional): The maximum noise level. Defaults to 80.
        rho (float, optional): The noise schedule exponent. Defaults to 7.0.
        time_power (float, optional): The power to which the timestep is raised in the MPNN schedule. Defaults to 4.0.
        constant_val (float, optional): The constant value for the constant schedule. Defaults to 0.0.

    Raises:
        ValueError: If the noise schedule function is unknown.

    Returns:
        torch.Tensor: The computed noise level.
    """

    if function == "lognormal":
        normal_sample = torch.tensor(norm.ppf(timestep.cpu())).to(timestep)
        noise_level = sigma_data * torch.exp(psigma_mean + psigma_std * normal_sample)
    elif function == "uniform":
        noise_level = compute_sampling_noise(
            timestep, sigma_data=sigma_data, s_min=s_min, s_max=s_max, rho=rho
        )
    elif function == "mpnn":
        timestep = timestep**time_power
        noise_level = compute_sampling_noise(
            timestep, sigma_data=sigma_data, s_min=s_min, s_max=s_max, rho=rho
        )
    elif function == "constant":
        noise_level = torch.ones_like(timestep) * constant_val
    else:
        raise ValueError(f"Unknown noise schedule function: {function}")

    return noise_level


def noise_coords(
    coords: torch.Tensor,
    noise_level: torch.Tensor,
    atom_mask: torch.Tensor,
    dummy_fill_mode: str = "zero",
) -> torch.Tensor:
    """Applies noise to the coordinates.

    Args:
        coords (torch.Tensor): The input coordinates. (B, N, A, X)
        noise_level (torch.Tensor): The noise level for each batch. (B,)
        atom_mask (torch.Tensor): The atom mask indicating which atoms are present. (B, N, A)

    Returns:
        torch.Tensor: The noisy coordinates. (B, N, A, X)
    """

    dummy_fill_mask = 1.0 - atom_mask
    if dummy_fill_mode == "zero":
        dummy_fill_value = torch.zeros_like(coords)
    else:
        dummy_fill_value = coords[..., 1:2, :]  # CA
    coords = (
        coords * atom_mask[..., None] + dummy_fill_value * dummy_fill_mask[..., None]
    )

    noise = torch.randn_like(coords) * unsqueeze_trailing_dims(noise_level, coords)
    noisy_coords = coords + noise

    return noisy_coords
