"""Dataloader from PDB files.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import math
from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch.utils.data import Dataset, RandomSampler, Sampler
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.configs import TrainingConfig
from protpardelle.data.atom import dummy_fill_noise_coords
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.utils import get_logger, unsqueeze_trailing_dims

logger = get_logger(__name__)

FEATURES_1D = (
    "coords_in",
    "torsions_in",
    "b_factors",
    "atom_positions",
    "aatype",
    "atom_mask",
    "residue_index",
    "chain_index",
    "fluctuation",
    "displacement",
    "sse",
)
FEATURES_2D = ("adj",)
FEATURES_FLOAT = (
    "coords_in",
    "torsions_in",
    "b_factors",
    "atom_positions",
    "atom_mask",
    "seq_mask",
    "fluctuation",
    "displacement",
)
FEATURES_LONG = ("aatype", "residue_index", "chain_index", "orig_size", "sse", "adj")


def make_fixed_size_1d(
    data: Float[torch.Tensor, "L ..."], fixed_size: int = 128
) -> tuple[Float[torch.Tensor, "N ..."], Float[torch.Tensor, "N"]]:
    """Pads or crops a 1D tensor.

    Args:
        data (torch.Tensor): Input tensor.
        fixed_size (int): Desired length.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            new_data (torch.Tensor): New tensor.
            mask (torch.Tensor): Binary mask indicating valid data.
    """

    data_len = data.shape[0]
    device = data.device

    if data_len >= fixed_size:
        extra_len = data_len - fixed_size
        start_idx = np.random.choice(np.arange(extra_len + 1))
        new_data = data[start_idx : (start_idx + fixed_size)]
        mask = torch.ones(fixed_size, device=device)
    else:
        pad_size = fixed_size - data_len
        extra_shape = data.shape[1:]
        new_data = torch.cat(
            [data, torch.zeros(pad_size, *extra_shape).to(data.device)], 0
        )
        mask = torch.cat(
            [torch.ones(data_len, device=device), torch.zeros(pad_size, device=device)],
            dim=0,
        )

    return new_data, mask


def make_fixed_size_2d(
    data: Float[torch.Tensor, "H W ..."], fixed_size: int = 128
) -> tuple[Float[torch.Tensor, "N N ..."], Float[torch.Tensor, "N N"]]:
    """Pads or crops a 2D tensor.

    Args:
        data (torch.Tensor): Input tensor.
        fixed_size (int, optional): Desired height and width. Defaults to 128.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            new_data (torch.Tensor): New tensor.
            mask (torch.Tensor): Binary mask indicating valid data.
    """

    H, W, *extra_shapes = data.shape
    device = data.device

    # Crop if larger than fixed_size
    if H >= fixed_size:
        h_start = np.random.randint(0, H - fixed_size + 1)
        h_end = h_start + fixed_size
        h_len = fixed_size
    else:
        h_start = 0
        h_end = H
        h_len = H

    if W >= fixed_size:
        w_start = np.random.randint(0, W - fixed_size + 1)
        w_end = w_start + fixed_size
        w_len = fixed_size
    else:
        w_start = 0
        w_end = W
        w_len = W

    cropped = data[h_start:h_end, w_start:w_end]

    # Build pad tuple from the LAST dimension backwards; prepend (0,0) for each trailing extra dim.
    # Final pairs correspond to (W, then H): (left_w, right_w, top_h, bottom_h) = (0, pad_w, 0, pad_h)
    pad_h = fixed_size - h_len
    pad_w = fixed_size - w_len
    pad_dims = (0, 0) * len(extra_shapes) + (
        0,
        pad_w,
        0,
        pad_h,
    )

    new_data = F.pad(cropped, pad_dims)

    # Mask is 2D (H, W), so pad with 4 values only (W then H).
    valid = torch.ones((h_len, w_len), device=device)
    pad_dims_mask = (0, pad_w, 0, pad_h)
    mask = F.pad(valid, pad_dims_mask)

    return new_data, mask


def apply_random_se3(
    atom_coords: Float[torch.Tensor, "L A 3"],
    atom_mask: Float[torch.Tensor, "L A"] | None = None,
    translation_scale: float = 1.0,
) -> Float[torch.Tensor, "L A 3"]:
    """Applies a random rotation and translation to the coordinates.

    Not batched.

    Args:
        atom_coords (torch.Tensor): Input coordinates.
        atom_mask (torch.Tensor | None, optional): Atom mask.
        translation_scale (float, optional): Scale for translation. Defaults to 1.0.

    Returns:
        torch.Tensor: Transformed coordinates.
    """

    coords_mean = atom_coords[:, 1:2].mean(dim=-3, keepdim=True)
    atom_coords = atom_coords - coords_mean

    random_rot = uniform_rand_rotation(1).squeeze(0)
    atom_coords = atom_coords @ random_rot
    random_trans = torch.randn_like(coords_mean) * translation_scale
    atom_coords = atom_coords + random_trans

    if atom_mask is not None:
        atom_coords = atom_coords * atom_mask.unsqueeze(-1)
    return atom_coords


def uniform_rand_rotation(
    batch_size: int, seed: int | None = None
) -> Float[torch.Tensor, "B 3 3"]:
    """Creates a rotation matrix uniformly at random in SO(3).

    Uses quaternionic multiplication to generate independent rotation matrices for each batch.

    Args:
        batch_size (int): The number of rotation matrices to generate.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        torch.Tensor: The generated rotation matrices.
    """

    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    q = torch.randn(batch_size, 4, generator=rng)
    q = q / torch.norm(q, dim=1, keepdim=True)
    rotation = torch.zeros(batch_size, 3, 3)

    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation[:, 0, :] = torch.stack(
        [2 * a**2 - 1 + 2 * b**2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c]
    ).t()
    rotation[:, 1, :] = torch.stack(
        [2 * b * c + 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d - 2 * a * b]
    ).t()
    rotation[:, 2, :] = torch.stack(
        [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, 2 * a**2 - 1 + 2 * d**2]
    ).t()

    return rotation


def get_masked_coords_array(
    atom_coords: Float[torch.Tensor, "L A 3"],
    atom_mask: Float[torch.Tensor, "L A"],
) -> np.ma.MaskedArray:
    """Create a masked array from atom coordinates and mask.

    Args:
        atom_coords (torch.Tensor): Atom coordinates.
        atom_mask (torch.Tensor): Atom mask.

    Returns:
        np.ma.MaskedArray: Masked array of atom coordinates.
    """

    ma_mask = repeat(1 - atom_mask.unsqueeze(-1).cpu().numpy(), "... 1 -> ... 3")

    return np.ma.array(atom_coords.cpu().numpy(), mask=ma_mask)


def calc_sigma_data(
    dataset: Dataset,
    config: TrainingConfig,
    num_workers: int,
) -> float:
    """Given a dataset and the model training config, estimate sigma_data.

    Args:
        dataset (Dataset): The dataset to use for estimation.
        config (TrainingConfig): The training configuration.
        num_workers (int): The number of workers to use for data loading.

    Returns:
        float: The estimated sigma_data.
    """

    sigma_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
    )

    collected_coords = []
    collected_atom_masks = []
    collected_seq_masks = []

    num_batches = math.ceil(
        config.data.n_examples_for_sigma_data / config.train.batch_size
    )
    for i, inputs in tqdm(
        enumerate(sigma_dataloader), desc="Collecting data", total=num_batches
    ):
        # Stop collecting once we've reached enough examples
        if i == num_batches:
            break

        seq_mask = inputs["seq_mask"]
        coords = inputs["coords_in"]
        atom_mask = inputs["atom_mask"]

        if config.train.crop_conditional:
            crop_cond_vars = {k: v for k, v in vars(config.train.crop_cond).items() if not k.startswith('_')}
            coords, _, _ = make_crop_cond_mask_and_recenter_coords(
                atom_coords=coords, atom_mask=atom_mask, **crop_cond_vars
            )

        collected_coords.append(coords)
        collected_atom_masks.append(atom_mask)
        collected_seq_masks.append(seq_mask)

    # Convert collected data lists to tensors
    coords = torch.cat(collected_coords, dim=0)[: config.data.n_examples_for_sigma_data]
    atom_mask = torch.cat(collected_atom_masks, dim=0)[
        : config.data.n_examples_for_sigma_data
    ]
    seq_mask = torch.cat(collected_seq_masks, dim=0)[
        : config.data.n_examples_for_sigma_data
    ]

    if config.model.compute_loss_on_all_atoms:
        # Compute sigma_data on all 37 atoms for each residue
        atom_mask = torch.ones_like(atom_mask) * unsqueeze_trailing_dims(
            seq_mask, atom_mask
        )

    # Estimate sigma_data
    masked_coords = get_masked_coords_array(coords, atom_mask)
    sigma_data = float(masked_coords.std())

    return sigma_data


def make_crop_cond_mask_and_recenter_coords(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    aatype: torch.Tensor | None = None,
    chain_index: torch.Tensor | None = None,
    contiguous_prob: float = 0.05,
    discontiguous_prob: float = 0.9,
    sidechain_prob: float = 0.9,
    sidechain_only_prob: float = 0.0,
    sidechain_tip_prob: float = 0.0,
    max_span_len: int = 10,
    max_discontiguous_res: int = 8,
    dist_threshold: float = 8.0,
    recenter_coords: bool = True,
    add_coords_noise: float = 0.0,
    multichain_prob: float = 0.0,
    hotspot_min: int = 3,
    hotspot_max: int = 8,
    hotspot_dropout: float = 0.1,
    paratope_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random motif crop from a batch of protein structures.

    Args:
        atom_mask (torch.Tensor): Binary mask where 1 indicates an existing atom
            and 0 otherwise. (B, N, A)
        atom_coords (torch.Tensor): Cartesian atom coordinates in atom37 order
            (typically A=37, C=3). (B, N, A, C)
        aatype (torch.Tensor | None, optional): Integer-encoded amino acid types.
            Defaults to None. (B, N)
        chain_index (torch.Tensor | None, optional): Integer-encoded chain IDs.
            Defaults to None. (B, N)
        contiguous_prob (float, optional): Probability of sampling a contiguous
            motif. Defaults to 0.05.
        discontiguous_prob (float, optional): Probability of sampling a
            discontiguous motif. Defaults to 0.9.
        sidechain_prob (float, optional): Probability of including sidechain
            coordinates in the motif information. Defaults to 0.9.
        sidechain_only_prob (float, optional): Probability of including only
            sidechain coordinates (mask backbone coordinates). Defaults to 0.0.
        sidechain_tip_prob (float, optional): Probability of including only
            sidechain tip atoms (mask backbone and non-tip sidechain atoms).
            Defaults to 0.0.
        max_span_len (int, optional): Maximum contiguous motif length that can be
            sampled. Defaults to 10.
        max_discontiguous_res (int, optional): Maximum number of discontiguous
            residues that can be sampled. Defaults to 8.
        dist_threshold (float, optional): Neighborhood distance threshold used to
            include additional residues for discontiguous motifs (in the same
            units as atom_coords). Defaults to 8.0.
        recenter_coords (bool, optional): If True, recenter the entire structure so
            the motif center of mass is at the origin. Defaults to True.
        add_coords_noise (float, optional): Standard deviation of Gaussian noise
            added to motif coordinates for regularization. Defaults to 0.0.
        multichain_prob (float, optional): Probability of using the multichain
            scaffolding task (motif is one entire chain; scaffold is the partner
            chain). Defaults to 0.0.
        hotspot_min (int, optional): Minimum number of hotspots for the multichain
            task. Defaults to 3.
        hotspot_max (int, optional): Maximum number of hotspots for the multichain
            task. Defaults to 8.
        hotspot_dropout (float, optional): Probability of dropping hotspot
            information (set to all zeros). Defaults to 0.1.
        paratope_prob (float, optional): Probability of including additional
            paratope residues together with the target chain as the motif.
            Defaults to 0.5.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - coords_out: Cropped (and optionally recentered/noised) coordinates.
            (B, N, A, C)
            - crop_cond_mask: Binary mask indicating atoms included in the motif
            crop/conditioning. (B, N, A)
            - hotspot_mask: Binary mask indicating hotspot atoms. (B, N, A)
    """

    b, n, a = atom_mask.shape
    device = atom_mask.device
    seq_mask = atom_mask[..., 1]
    num_res = seq_mask.sum(-1)
    masks = []
    all_hotspot_masks = torch.zeros((b, n), device=device)

    for i, nr in enumerate(num_res):
        nr = nr.int().item()
        mask = torch.zeros((n, a), device=device)
        if chain_index is not None and chain_index[i].sum(-1) > 0:
            conditioning_type = torch.distributions.Categorical(
                torch.tensor(
                    [
                        contiguous_prob,
                        discontiguous_prob,
                        multichain_prob,
                        1.0 - contiguous_prob - discontiguous_prob - multichain_prob,
                    ]
                )
            ).sample()
            conditioning_type = ["contiguous", "discontiguous", "multichain", "none"][
                conditioning_type
            ]
        else:
            conditioning_type = torch.distributions.Categorical(
                torch.tensor(
                    [
                        0.05,
                        0.9,
                        0.05,
                    ]
                )
            ).sample()
            conditioning_type = ["contiguous", "discontiguous", "none"][
                conditioning_type
            ]

        if conditioning_type == "contiguous":
            span_len = torch.randint(
                1, min(max_span_len, nr), (1,), device=device
            ).item()
            span_start = torch.randint(0, nr - span_len, (1,), device=device)
            mask[span_start : span_start + span_len, :] = 1
        elif conditioning_type == "discontiguous":
            # Extract CB atoms coordinates for the i-th example
            cb_atoms = atom_coords[i, :, 3]
            # Pairwise distances between CB atoms
            cb_distances = torch.cdist(cb_atoms, cb_atoms)
            close_mask = (
                cb_distances <= dist_threshold
            )  # Mask for selecting close CB atoms

            random_residue = torch.randint(0, nr, (1,), device=device).squeeze()
            cb_dist_i = cb_distances[random_residue] + 1e3 * (1 - seq_mask[i])
            close_mask = cb_dist_i <= dist_threshold
            num_neighbors = close_mask.sum().int()

            # pick how many neighbors (up to 10)
            num_sele = torch.randint(
                2,
                num_neighbors.clamp(min=3, max=max_discontiguous_res + 1),
                (1,),
                device=device,
            )

            # Select the indices of CB atoms that are close together
            idxs = torch.arange(n, device=device)[close_mask.bool()]
            idxs = idxs[torch.randperm(len(idxs))[:num_sele]]

            if len(idxs) > 0:
                mask[idxs] = 1
        # Keep one chain as the motif, generate the other chain.
        elif conditioning_type == "multichain":  # check if actually multichain
            chain_as_motif = np.random.choice(torch.unique(chain_index[i]).cpu())
            idx_as_motif = (chain_index[i] == chain_as_motif) & (seq_mask[i] != 0)
            mask[idx_as_motif] = 1

            # generate hotspot input (which idx_as_motif are closest to the chain to generate)
            ca_atoms = atom_coords[i, :, 1]
            ca_distances = torch.cdist(ca_atoms, ca_atoms)
            x = chain_index[i]
            chain_dist = (x[:, None] - x[None, :]).abs()
            intrachain_mask = chain_dist == 0
            intrachain_mask[nr:] = 1
            intrachain_mask[:, nr:] = 1
            ca_distances[intrachain_mask] = 999.0
            flat_indices = torch.argsort(ca_distances.flatten())
            ii, jj = torch.unravel_index(flat_indices, chain_dist.shape)
            contact_idx_pairs = list(zip(ii.tolist(), jj.tolist()))
            num_hotspots = np.random.choice(np.arange(hotspot_min * 2, hotspot_max * 2))

            # randomly drop out hotspot 10% of the time
            if np.random.rand() < 1 - hotspot_dropout:
                for pair in contact_idx_pairs[:num_hotspots]:
                    idx_hotspot = min(pair) if chain_as_motif == 0 else max(pair)
                    all_hotspot_masks[i, idx_hotspot] = 1

            # include some paratope residues as part of the motif
            if np.random.rand() < paratope_prob:
                for pair in contact_idx_pairs[:num_hotspots]:
                    idx_paratope = max(pair) if chain_as_motif == 0 else min(pair)
                    flanking_width = np.random.choice(np.arange(0, 6))
                    mask[
                        max(
                            torch.sum(idx_as_motif).item(),
                            idx_paratope - flanking_width,
                        ) : min(idx_paratope + flanking_width, nr)
                    ] = 1

        if np.random.uniform() < sidechain_prob:  # keep all crop-cond coords unmasked
            if np.random.uniform() < sidechain_only_prob:
                mask[:, (0, 1, 2, 4)] = 0
            if np.random.uniform() < sidechain_tip_prob and aatype is not None:
                # determine tip atoms by amino acid type
                motif_idx = torch.nonzero(mask.sum(-1)).flatten()
                for mi in motif_idx:
                    aatype_int = aatype[i][mi]
                    motif_aatype_str = residue_constants.restype_1to3[
                        residue_constants.order_restype[aatype_int.item()]
                    ]

                    # Original Protpardelle tip atom definition, also used in La-Proteina
                    tip_atomtypes = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[
                        motif_aatype_str
                    ]
                    tip_atom_idx_atom37 = [
                        residue_constants.atom_order.get(atype)
                        for atype in tip_atomtypes
                    ]

                    nontip_idx = np.delete(np.arange(37), tip_atom_idx_atom37)
                    mask[mi, nontip_idx] = 0

        else:  # discard everything that is not backbone N, CA, C, O
            mask[:, 3] = 0
            mask[:, 5:] = 0

        masks.append(mask)

    crop_cond_mask = torch.stack(masks)
    crop_cond_mask = crop_cond_mask * atom_mask

    if recenter_coords:
        motif_masked_array = get_masked_coords_array(atom_coords, crop_cond_mask)
        cond_coords_center = motif_masked_array.mean((1, 2))
        motif_mask = torch.tensor(1 - cond_coords_center.mask).to(crop_cond_mask)
        means = torch.tensor(cond_coords_center.data).to(atom_coords) * motif_mask
        coords_out = atom_coords - rearrange(means, "b c -> b 1 1 c")
    else:
        coords_out = atom_coords

    if add_coords_noise > 0:
        coords_out = coords_out + add_coords_noise * torch.randn_like(coords_out)

    return coords_out, crop_cond_mask, all_hotspot_masks


class PDBDataset(Dataset):
    """Loads and processes PDBs into tensors."""

    def __init__(
        self,
        pdb_path: str,
        fixed_size: int,
        mode: str = "train",
        overfit: int = 0,
        short_epoch: bool = False,
        se3_data_augment: bool = True,
        translation_scale: float = 1.0,
        chain_residx_gap: int = 200,
        dummy_fill_mode: Literal["CA", "zero"] = "zero",
        subset: str | float = "",
    ) -> None:
        """Initialize the PDBDataset.

        Args:
            pdb_path (str): Path to the input PDB file.
            fixed_size (int): Target length used to trim or pad per-example tensors.
            mode (str, optional): Operating mode, either "train" or "eval".
                Defaults to "train".
            overfit (int, optional): For debugging: if > 0, restrict the dataset
                to a small subset of this size to intentionally overfit. Defaults to 0.
            short_epoch (bool, optional): For debugging: if True, stop an epoch early
                to shorten iteration time. Defaults to False.
            se3_data_augment (bool, optional): Apply random SE(3) rotation and
                translation to input coordinates. Defaults to True.
            translation_scale (float, optional): Standard deviation of the Gaussian
                translational perturbation (in the same units as coordinates).
                Defaults to 1.0.
            chain_residx_gap (int, optional): Offset added to residue indices to
                separate chains. Defaults to 200.
            dummy_fill_mode (Literal["CA", "zero"], optional): Strategy to fill coordinates
                for non-existing atoms. Defaults to "zero".
            subset (str | float, optional): Dataset-specific subset identifier to
                train on; if a float in (0, 1], interpreted as a fraction of data
                to sample. Defaults to "".
        """

        self.pdb_path = pdb_path
        self.fixed_size = fixed_size
        self.mode = mode
        self.overfit = overfit
        self.short_epoch = short_epoch
        self.se3_data_augment = se3_data_augment
        self.translation_scale = translation_scale
        self.chain_residx_gap = chain_residx_gap
        self.dummy_fill_mode = dummy_fill_mode

        if self.pdb_path.endswith("test_boltz_interfaces"):
            self.cif_df = pd.read_csv(f"{self.pdb_path}/test_interface_info.csv")
            self.pdb_keys = list(
                set(
                    self.cif_df["chain_1_cluster_id"].to_list()
                    + self.cif_df["chain_2_cluster_id"].to_list()
                )
            )
        elif self.pdb_path.endswith("boltz_interfaces"):
            self.cif_df = pd.read_csv(f"{self.pdb_path}/interface_info.csv")
            if isinstance(subset, float):
                orig_size = len(self.cif_df)
                self.cif_df = self.cif_df[self.cif_df["resolution"] <= subset]
                self.cif_df["total_length"] = (
                    self.cif_df["chain_1_num_residues"]
                    + self.cif_df["chain_2_num_residues"]
                )
                self.cif_df = self.cif_df[
                    self.cif_df["total_length"] <= self.fixed_size
                ]
                self.cif_df["homodimer"] = (
                    self.cif_df["chain_1_num_residues"]
                    == self.cif_df["chain_2_num_residues"]
                )
                self.cif_df = self.cif_df[
                    ~self.cif_df["homodimer"]
                ]  # filter out homodimers
                new_size = len(self.cif_df)
                logger.info(
                    "Filtering by resolution better than %sA and total length less than %s residues. Removed %s examples, will train on %s examples.",
                    subset,
                    self.fixed_size,
                    orig_size - new_size,
                    new_size,
                )
            self.pdb_keys = list(
                set(
                    self.cif_df["chain_1_cluster_id"].to_list()
                    + self.cif_df["chain_2_cluster_id"].to_list()
                )
            )
        else:
            with open(f"{self.pdb_path}/{mode}_{subset}_pdb_keys.list", "r") as f:
                self.pdb_keys = np.array(f.read().split("\n")[:-1])

        if overfit > 0:
            num_data = len(self.pdb_keys)
            self.pdb_keys = np.random.choice(
                self.pdb_keys, min(num_data, overfit), replace=False
            ).repeat(num_data // overfit)

    def __len__(self):
        return min(len(self.pdb_keys), 256) if self.short_epoch else len(self.pdb_keys)

    def __getitem__(self, idx):
        pdb_key = self.pdb_keys[idx]
        data = self.get_item(pdb_key)
        # For now, replace dataloading errors with a random pdb. 50 tries
        for _ in range(50):
            if data is not None:
                return data
            pdb_key = self.pdb_keys[np.random.randint(len(self.pdb_keys))]
            data = self.get_item(pdb_key)
        raise ValueError("Failed to load data example after 50 tries.")

    def get_item(self, pdb_key):
        example = {}

        chain_id = None
        if self.pdb_path.endswith("cath_s40_dataset"):  # CATH pdbs
            data_file = f"{self.pdb_path}/dompdb/{pdb_key}"
        elif "ingraham_cath_dataset" in self.pdb_path:  # ingraham splits
            data_file = f"{self.pdb_path}/pdb_store/{pdb_key}"
        elif self.pdb_path.endswith(
            "augmented_ingraham_cath_bugfree"
        ):  # mpnn augmented ingraham splits
            data_file = f"{self.pdb_path}/mpnn_esmfold/{pdb_key}"
            if not Path(data_file).exists():
                data_file = f"{self.pdb_path}/dne_mpnn/{pdb_key}"
        elif self.pdb_path.endswith("boltz_interfaces") or self.pdb_path.endswith(
            "test_boltz_interfaces"
        ):  # boltz interfaces curated by Richard
            df_subset = self.cif_df[
                (self.cif_df["chain_1_cluster_id"] == pdb_key)
                | (self.cif_df["chain_2_cluster_id"] == pdb_key)
            ]
            if df_subset.empty:
                raise ValueError(f"Cluster {pdb_key} does not exist!")
            pdb = df_subset.sample(n=1)["pdb_name"].values[0]
            data_file = f"{self.pdb_path}/interface_cifs/{pdb}"
        else:
            raise ValueError("Invalid pdb path.")

        try:
            example, _ = load_feats_from_pdb(
                data_file, chain_residx_gap=self.chain_residx_gap, chain_id=chain_id
            )
            coords_in = example["atom_positions"]
        except Exception as e:
            logger.error("Error loading PDB %s: %s", data_file, str(e))
            return

        # Apply data augmentation
        if self.se3_data_augment:
            coords_in = apply_random_se3(
                coords_in,
                atom_mask=example["atom_mask"],
                translation_scale=self.translation_scale,
            )
        else:
            coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)
            coords_in = coords_in - coords_mean

        ss_adj_dir = Path(self.pdb_path) / "ss_adj"
        sse_path = ss_adj_dir / f"{Path(data_file).stem}_ss.pt"
        adj_path = ss_adj_dir / f"{Path(data_file).stem}_adj.pt"
        if sse_path.exists() and adj_path.exists():
            example["sse"] = (
                torch.from_numpy(torch.load(sse_path, weights_only=False))
                .long()
                .to(example["residue_index"])
            )
            example["adj"] = (
                torch.load(adj_path, weights_only=False)
                .long()
                .to(example["residue_index"])
            )
        else:
            seqlen = example["residue_index"].shape[0]
            example["sse"] = torch.zeros_like(
                example["residue_index"], dtype=torch.long
            )
            example["adj"] = torch.zeros(seqlen, seqlen, dtype=torch.long).to(
                example["sse"]
            )

        if self.dummy_fill_mode != "zero":
            coords_in = dummy_fill_noise_coords(
                coords_in, example["atom_mask"], dummy_fill_mode=self.dummy_fill_mode
            )

        orig_size = coords_in.shape[0]
        if orig_size < 5:
            return None
        example["coords_in"] = coords_in
        example["orig_size"] = torch.ones(1) * orig_size

        fixed_size_example = {}
        seq_mask = None
        for k, v in example.items():
            if k in FEATURES_1D:
                fixed_size_example[k], seq_mask = make_fixed_size_1d(
                    v, fixed_size=self.fixed_size
                )
            elif k in FEATURES_2D:
                fixed_size_example[k], _ = make_fixed_size_2d(
                    v, fixed_size=self.fixed_size
                )
            else:
                fixed_size_example[k] = v

        if seq_mask is not None:
            fixed_size_example["seq_mask"] = seq_mask

        example_out = {}
        for k, v in fixed_size_example.items():
            if k in FEATURES_FLOAT:
                example_out[k] = v.float()
            elif k in FEATURES_LONG:
                example_out[k] = v.long()

        return example_out

    def collate(self, example_list):
        out = {}
        for ex in example_list:
            for k, v in ex.items():
                out.setdefault(k, []).append(v)
        return {k: torch.stack(v) for k, v in out.items()}

    def sample(self, n=1, return_data=True, return_keys=False):
        keys = self.pdb_keys[torch.randperm(self.__len__())[:n].long()]

        if return_keys and not return_data:
            return keys

        if n == 1:
            data = self.collate([self.get_item(keys)])
        else:
            data = self.collate([self.get_item(key) for key in keys])

        if return_data and return_keys:
            return data, keys
        if return_data and not return_keys:
            return data


class StochasticMixedSampler(Sampler):
    """Stochastic Mixed Sampler.

    A sampler to draw samples from multiple datasets. This sampler is specifically designed
    to accommodate a setup where there's one primary dataset that needs to be fully iterated
    without replacement (typically primary examples) and several augmented datasets where samples
    can be drawn with replacement. The ratio of drawing from each dataset is determined by the
    provided mixing ratios, but we are guaranteed to draw a fixed number of samples from the
    primary dataset (equal to int(batch_size * mixing_ratios[0])).

    Attributes:
        primary_dataset_length (int): The length of the primary dataset.
        primary_samples_per_batch (int): The number of primary dataset examples we draw per batch.
        offsets (list[int]): The accumulated offset of each dataset when all datasets are concatenated.
        samplers (list[Sampler]): A list of samplers for each dataset. The primary dataset uses a
            RandomSampler without replacement while all others use a RandomSampler with replacement.
    """

    def __init__(
        self,
        datasets: Sequence[PDBDataset],
        mixing_ratios: list[float],
        batch_size: int,
    ) -> None:
        """Stochastic Mixed Sampler.

        Args:
            datasets (Sequence[PDBDataset]): A list of datasets to sample from. The first dataset
                is considered the primary dataset that will be sampled without replacement.
            mixing_ratios (list[float]): A list of floats representing the mixing ratio for each dataset.
                The sum of all ratios should be equal to 1. The length of this list should be equal
                to the number of datasets.
            batch_size (int): The batch size for which samples need to be drawn.
        """

        self.datasets = datasets
        self.mixing_ratios = np.array(mixing_ratios)
        self.batch_size = batch_size
        self.primary_dataset_length = len(datasets[0])

        # Deterministic number of primary examples to draw per batch
        self.primary_samples_per_batch = int(batch_size * mixing_ratios[0])
        self.offsets = [0] + list(accumulate(len(dataset) for dataset in datasets[:-1]))

        assert (
            sum(mixing_ratios) == 1
        ), "Mixing ratios for drawing samples from datasets do not sum to 1."

        # Samplers
        self.samplers = [
            RandomSampler(datasets[0], replacement=False)
        ]  # primary dataset without replacement
        self.samplers.extend(
            [RandomSampler(dataset, replacement=True) for dataset in datasets[1:]]
        )  # Augmented datasets with replacement

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]

        while True:
            batch_indices = []

            # Draw the fixed number of primary dataset examples
            for _ in range(self.primary_samples_per_batch):
                primary_sample = next(iterators[0], None)
                if primary_sample is None:  # If primary dataset is exhausted
                    return  # This will end the iterator
                batch_indices.append(primary_sample + self.offsets[0])

            # Calculate the remaining size of the batch
            remaining_size = self.batch_size - self.primary_samples_per_batch
            if remaining_size > 0:
                # Randomly determine number of instances from the augmented datasets for this batch
                num_samples = np.random.multinomial(
                    remaining_size, self.mixing_ratios[1:] / sum(self.mixing_ratios[1:])
                )

                for i, num in enumerate(num_samples, start=1):
                    batch_indices.extend(
                        [
                            next(iterators[i], None) + self.offsets[i]
                            for _ in range(num)
                            if next(iterators[i], None) is not None
                        ]
                    )

            # Yield the indices for this batch
            yield from batch_indices

    def __len__(self) -> int:
        return (
            int(np.ceil(len(self.datasets[0]) / self.primary_samples_per_batch))
            * self.batch_size
        )
