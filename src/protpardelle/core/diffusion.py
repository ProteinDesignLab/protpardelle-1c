"""Noise and diffusion utils.

Authors: Alex Chu, Tianyu Lu, Zhaoyang Li
"""

from typing import Literal

import torch
from jaxtyping import Float

from protpardelle.data.atom import dummy_fill
from protpardelle.utils import unsqueeze_trailing_dims


def compute_sampling_noise(
    timestep: Float[torch.Tensor, "..."],
    sigma_data: float = 10.0,
    s_min: float = 0.001,
    s_max: float = 80.0,
    rho: float = 7.0,
) -> Float[torch.Tensor, "..."]:
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
    timestep: Float[torch.Tensor, "..."],
    function: Literal["uniform", "lognormal", "mpnn", "constant"] = "uniform",
    sigma_data: float = 10.0,
    psigma_mean: float = -0.5,
    psigma_std: float = 1.5,
    s_min: float = 0.001,
    s_max: float = 80.0,
    rho: float = 7.0,
    time_power: float = 4.0,
    constant_val: float = 0.0,
) -> Float[torch.Tensor, "..."]:
    """Computes the noise schedule for a given timestep.

    Args:
        timestep (torch.Tensor): The current timestep.
        function (Literal["uniform", "lognormal", "mpnn", "constant"], optional):
            The noise schedule function to use. Defaults to "uniform".
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
        normal_sample = torch.special.ndtri(timestep)  # pylint: disable=not-callable
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
        noise_level = torch.full_like(timestep, constant_val)
    else:
        raise ValueError(f"Unknown noise schedule function: {function}")

    return noise_level


def noise_coords(
    atom37_coords: Float[torch.Tensor, "B L 37 3"],
    atom37_mask: Float[torch.Tensor, "B L 37"],
    noise_level: Float[torch.Tensor, "B"],
    dummy_fill_mode: Literal["CA", "zero"] = "zero",
) -> Float[torch.Tensor, "B L 37 3"]:
    """Applies noise to the coordinates.

    Args:
        atom37_coords (torch.Tensor): The input coordinates.
        atom37_mask (torch.Tensor): The atom mask indicating which atoms are present.
        noise_level (torch.Tensor): The noise level for each batch.
        dummy_fill_mode (Literal["CA", "zero"], optional): The mode for filling in dummy atoms.
            Defaults to "zero".

    Returns:
        torch.Tensor: The noisy coordinates.
    """

    atom37_coords = dummy_fill(
        atom37_coords, atom37_mask, mode=dummy_fill_mode
    )  # (B, L, 37, 3)

    noise = torch.randn_like(atom37_coords) * unsqueeze_trailing_dims(
        noise_level, target=atom37_coords
    )
    noisy_coords = atom37_coords + noise

    return noisy_coords
