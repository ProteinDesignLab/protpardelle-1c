"""Miscellaneous utils.

Authors: Alex Chu, Zhaoyang Li, Tianyu Lu
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
import yaml
from torchtyping import TensorType

if TYPE_CHECKING:
    from _typeshed import StrPath
else:
    StrPath: TypeAlias = str | os.PathLike[str]


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


def seed_everything(seed: int = 42) -> None:
    """Set the seed for all random number generators.

    Args:
        seed (int, optional): The seed value. Defaults to 42.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def unsqueeze_trailing_dims(x, target=None, add_ndims=1):
    if target is None:
        for _ in range(add_ndims):
            x = x[..., None]
    else:
        while len(x.shape) < len(target.shape):
            x = x[..., None]
    return x


def check_nan_inf(x):
    return torch.isinf(x).sum() + torch.isnan(x).sum()


def hook_fn(name, verbose=False):
    def f(grad):
        if check_nan_inf(grad) > 0:
            print(name, "grad nan/infs", grad.shape, check_nan_inf(grad), grad)
        if verbose:
            print(name, "grad shape", grad.shape, "norm", grad.norm())

    return f


def trigger_nan_check(name, x):
    if check_nan_inf(x) > 0:
        print(name, check_nan_inf(x))
        raise Exception


def directory_find(atom, root="."):
    for path, dirs, files in os.walk(root):
        if atom in dirs:
            return os.path.join(path, atom)


def dict_to_namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(path, return_dict=False):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    if return_dict:
        return config, config_dict
    else:
        return config


def parse_fixed_pos_str(
    fixed_pos_str: str,
    chain_id_mapping: dict[str, int],
    residue_index: TensorType["n", int],
    chain_index: TensorType["n", int],
) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A1", "A10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
        chain_id_mapping (dict): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices (shape: [N]).
        chain_index (torch.Tensor): Tensor of chain indices (shape: [N]).

    Returns:
        List[int]: List of absolute indices to set to 1 in the masks.
    """
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1)
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

        if chain_letter not in chain_id_mapping:
            raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

        # For the given chain, create a mask for all residues in the desired range
        chain_i = chain_id_mapping[chain_letter]
        range_mask = (
            (chain_index == chain_i)
            & (residue_index >= start_residue)
            & (residue_index <= end_residue)
        )
        matching_indices = torch.where(range_mask)[0]

        # Check that each residue in the requested range; warn if not found
        found_residues = residue_index[matching_indices].tolist()
        found_residues_set = set(found_residues)

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues_set:
                print(
                    f"Warning: Requested position {chain_letter}{r} not found in structure."
                )

        # Extend our fixed indices with whatever we did find
        fixed_indices.extend(matching_indices.tolist())

    return fixed_indices
