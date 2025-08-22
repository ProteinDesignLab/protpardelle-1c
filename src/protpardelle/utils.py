"""Miscellaneous utils.

Authors: Alex Chu, Zhaoyang Li, Tianyu Lu
"""

import argparse
import os
import random
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from _typeshed import StrPath
else:
    StrPath: TypeAlias = str | os.PathLike[str]


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


def load_config(path: StrPath) -> argparse.Namespace:
    """Load a YAML configuration file and convert it to a namespace."""

    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    return config


def norm_path(
    path: StrPath,
    *,
    expandvars: bool = True,
    expanduser: bool = True,
    resolve: bool = True,
) -> Path:
    """Normalize a file path.

    Args:
        path (StrPath): The file path to normalize.
        expandvars (bool, optional): Whether to expand environment variables. Defaults to True.
        expanduser (bool, optional): Whether to expand the user directory. Defaults to True.
        resolve (bool, optional): Whether to resolve the path. Defaults to True.

    Returns:
        Path: The normalized file path.
    """

    p = Path(path)
    if expandvars:
        p = Path(os.path.expandvars(p))
    if expanduser:
        p = p.expanduser()
    if resolve:
        p = p.resolve()

    return p


def seed_everything(seed: int = 0) -> None:
    """Set the seed for all random number generators.

    Args:
        seed (int, optional): The seed value. Defaults to 0.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
