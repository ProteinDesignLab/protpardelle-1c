"""Miscellaneous utils.

Authors: Alex Chu, Zhaoyang Li, Tianyu Lu
"""

import argparse
import random
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import torch
import yaml
from torch.types import Device

from protpardelle.core.models import Protpardelle
from protpardelle.env import StrPath, norm_path


class DotDict(dict):
    """A dictionary that supports dot notation access to its attributes.

    This class extends the built-in dict to allow accessing dictionary keys
    using dot notation (e.g., obj.key instead of obj['key']).
    """

    def __getattr__(self, key):
        """Get an attribute using dot notation."""
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from e

    def __setattr__(self, key, value):
        """Set an attribute using dot notation."""
        self[key] = value

    def __delattr__(self, key):
        """Delete an attribute using dot notation."""
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from e


def apply_dotdict_recursively(input_obj: Any) -> Any:
    """Convert dictionaries to DotDict instances recursively.

    Args:
        input_obj (Any): The input object to process. Can be a dictionary, list,
            or any other type.

    Returns:
        Any: The processed object with all dictionaries converted to DotDict instances.
            Non-dictionary objects are returned unchanged.
    """

    if input_obj is None:
        return None
    if isinstance(input_obj, dict):
        # Convert the current dictionary to a dotdict
        return DotDict({k: apply_dotdict_recursively(v) for k, v in input_obj.items()})
    if isinstance(input_obj, list):
        # Apply recursively to all elements in the list
        return [apply_dotdict_recursively(item) for item in input_obj]

    # Return the object as-is if it's neither a dict nor a list
    return input_obj


def clean_gpu_cache(func: Callable) -> Callable:
    """Decorator to clean GPU memory cache after the decorated function is executed."""

    counter = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal counter
        try:
            result = func(*args, **kwargs)
        finally:
            # gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                counter += 1
        return result

    return wrapper


def dict_to_namespace(config: dict) -> argparse.Namespace:
    """Convert a dictionary to a namespace recursively."""

    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)

    return namespace


def get_default_device() -> torch.device:
    """Get the default device for PyTorch tensors.

    Returns:
        torch.device: The default device (CPU or GPU).
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_config(config_path: StrPath) -> argparse.Namespace:
    """Load a YAML configuration file and convert it to a namespace."""
    config_path = norm_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    return config


def load_model(
    config_path: StrPath, checkpoint_path: StrPath, device: Device = None
) -> Protpardelle:
    """Load a Protpardelle model from a configuration file and a checkpoint."""
    if device is None:
        device = get_default_device()
    assert isinstance(device, torch.device)  # for mypy
    config = load_config(config_path)

    checkpoint_path = norm_path(checkpoint_path)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )["model_state_dict"]

    model = Protpardelle(config, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def seed_everything(seed: int = 0, freeze_cuda: bool = False) -> None:
    """Set the seed for all random number generators.
    Freeze CUDA for reproducibility if needed.

    Args:
        seed (int, optional): The seed value. Defaults to 0.
        freeze_cuda (bool, optional): Whether to freeze CUDA for reproducibility. Defaults to False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if freeze_cuda:
        # nonrandom CUDNN convolution algo, maybe slower
        torch.backends.cudnn.deterministic = True
        # nonrandom selection of CUDNN convolution, maybe slower
        torch.backends.cudnn.benchmark = False


def unsqueeze_trailing_dims(
    x: torch.Tensor, target: torch.Tensor | None = None, add_ndims: int = 1
) -> torch.Tensor:
    """Unsqueeze the trailing dimensions of a tensor.

    Args:
        x (torch.Tensor): The input tensor.
        target (torch.Tensor | None, optional): The target tensor to match dimensions with. Defaults to None.
        add_ndims (int, optional): The number of dimensions to add. Defaults to 1.

    Returns:
        torch.Tensor: The modified tensor with trailing dimensions unsqueezed.
    """

    if target is None:
        for _ in range(add_ndims):
            x = x[..., None]
    else:
        while len(x.shape) < len(target.shape):
            x = x[..., None]

    return x
