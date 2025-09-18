"""Miscellaneous utils.

Authors: Alex Chu, Zhaoyang Li, Tianyu Lu
"""

import argparse
import logging
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


def apply_dotdict_recursively(input_obj: dict[str, Any]) -> DotDict:
    """Convert dictionaries to DotDict instances recursively.

    Args:
        input_obj (Any): The input object to process. Can be a dictionary, list,
            or any other type.

    Returns:
        Any: The processed object with all dictionaries converted to DotDict instances.
            Non-dictionary objects are returned unchanged.
    """

    if isinstance(input_obj, dict):
        for key in input_obj:
            if not isinstance(key, str):
                raise TypeError(
                    f"Non-string key detected in dictionary: {key} (type: {type(key)})"
                )
            if key in dict.__dict__:
                raise KeyError(
                    f"Key '{key}' in dictionary shadows a built-in dict attribute."
                )
        # Convert the current dictionary to a dotdict
        return DotDict({k: apply_dotdict_recursively(v) for k, v in input_obj.items()})

    # Return the object as-is for non-dicts (lists, scalars, etc.)
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


def dict_to_namespace(d: dict) -> argparse.Namespace:
    """Convert a dictionary to a namespace recursively."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)

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


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with the specified name and level.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO (20).

    Returns:
        logging.Logger: The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def load_config(config_path: StrPath) -> argparse.Namespace:
    """Load a YAML configuration file and convert it to a namespace."""
    config_path = norm_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)

    return config


def namespace_to_dict(namespace: argparse.Namespace) -> dict:
    """Convert a namespace to a dictionary recursively."""
    d = {}
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            d[key] = namespace_to_dict(value)
        else:
            d[key] = value

    return d


def norm_path(
    path: StrPath,
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


def seed_everything(seed: int = 0, freeze_cuda: bool = False) -> None:
    """Set the seed for all random number generators.

    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/seed.py
    Freeze CUDA for reproducibility if needed.

    Args:
        seed (int, optional): The seed value. Defaults to 0.
        freeze_cuda (bool, optional): Whether to freeze CUDA for reproducibility. Defaults to False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if freeze_cuda:
        # nonrandom CUDNN convolution algo, maybe slower
        torch.backends.cudnn.deterministic = True
        # nonrandom selection of CUDNN convolution, maybe slower
        torch.backends.cudnn.benchmark = False


def tensor_to_ndarray(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(
        f"Expected input to be a torch.Tensor or np.ndarray, but got {type(x)}"
    )


def unsqueeze_trailing_dims(
    x: torch.Tensor, target: torch.Tensor | None = None, add_ndims: int = 1
) -> torch.Tensor:
    """Unsqueeze the trailing dimensions of a tensor.

    Args:
        x (torch.Tensor): The input tensor.
        target (torch.Tensor | None, optional): The target tensor to match dimensions with.
            If None, add_ndims will be used. Defaults to None.
        add_ndims (int, optional): The number of dimensions to add. Can be overridden by target.
            Defaults to 1.

    Returns:
        torch.Tensor: The modified tensor with trailing dimensions unsqueezed.
    """

    if target is not None:
        add_ndims = target.ndim - x.ndim

    if add_ndims > 0:
        return x[(...,) + (None,) * add_ndims]

    raise ValueError("Must add a positive number of dimensions.")
