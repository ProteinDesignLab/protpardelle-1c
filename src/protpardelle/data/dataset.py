"""Dataloader from PDB files.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import argparse
import math
from itertools import accumulate
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset, RandomSampler, Sampler
from torchtyping import TensorType
from tqdm.auto import tqdm

import protpardelle.data.pdb_io
import protpardelle.utils
from protpardelle.common.residue_constants import (
    RFDIFFUSION_BENCHMARK_TIP_ATOMS,
    atom_order,
    order_restype,
    restype_1to3,
)
from protpardelle.data import atom

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


def make_fixed_size_1d(data, fixed_size=128):
    """
    Pads or crops a 1D tensor (L x ...) to fixed_size x ...

    Args:
        data (torch.Tensor): Input tensor of shape (L, ...).
        fixed_size (int): Desired length.

    Returns:
        new_data (torch.Tensor): Tensor of shape (fixed_size, ...).
        mask (torch.Tensor): Binary mask of shape (fixed_size) indicating valid data.
    """
    data_len = data.shape[0]
    if data_len >= fixed_size:
        extra_len = data_len - fixed_size
        start_idx = np.random.choice(np.arange(extra_len + 1))
        new_data = data[start_idx : (start_idx + fixed_size)]
        mask = torch.ones(fixed_size)
    else:
        pad_size = fixed_size - data_len
        extra_shape = data.shape[1:]
        new_data = torch.cat(
            [data, torch.zeros(pad_size, *extra_shape).to(data.device)], 0
        )
        mask = torch.cat([torch.ones(data_len), torch.zeros(pad_size)], 0)
    return new_data, mask


def make_fixed_size_2d(data, fixed_size=128):
    """
    Pads or crops a 2D tensor (H x W x ...) to fixed_size x fixed_size, ...

    Args:
        data (torch.Tensor): Input tensor of shape (H, W, ...).
        fixed_size (int): Desired height and width.

    Returns:
        new_data (torch.Tensor): Tensor of shape (fixed_size, fixed_size, ...).
        mask (torch.Tensor): Binary mask of shape (fixed_size, fixed_size) indicating valid data.
    """
    H, W = data.shape[:2]
    extra_shape = data.shape[2:]

    # Crop if larger than fixed_size
    if H >= fixed_size:
        h_start = np.random.randint(0, H - fixed_size + 1)
        h_end = h_start + fixed_size
    else:
        h_start, h_end = 0, H

    if W >= fixed_size:
        w_start = np.random.randint(0, W - fixed_size + 1)
        w_end = w_start + fixed_size
    else:
        w_start, w_end = 0, W

    cropped = data[h_start:h_end, w_start:w_end]

    # Pad if smaller than fixed_size
    pad_h = fixed_size - (h_end - h_start)
    pad_w = fixed_size - (w_end - w_start)

    pad_dims = (0, 0) * len(extra_shape) + (
        0,
        pad_w,
        0,
        pad_h,
    )  # (W_right, W_left, H_bottom, H_top)
    new_data = F.pad(cropped, pad_dims)

    return new_data


def apply_random_se3(
    coords_in, atom_mask=None, translation_scale=1.0, return_rot: bool = False
):
    # unbatched. center on the mean of CA coords
    coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)
    coords_in -= coords_mean
    random_rot = uniform_rand_rotation(1).squeeze(0)
    coords_in = coords_in @ random_rot
    random_trans = torch.randn_like(coords_mean) * translation_scale
    coords_in += random_trans
    if atom_mask is not None:
        coords_in = coords_in * atom_mask.unsqueeze(-1)
    if return_rot:
        return coords_in, random_rot
    else:
        return coords_in


def uniform_rand_rotation(batch_size):
    """
    Creates a shape (batch_size, 3, 3) rotation matrix uniformly at random in SO(3)
    Uses quaternionic multiplication to generate independent rotation matrices for each batch

    Credit: Steven Dunne
    """
    q = torch.randn(batch_size, 4)
    q /= torch.norm(q, dim=1, keepdim=True)
    rotation = torch.zeros(batch_size, 3, 3).to(q)
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation[:, 0, :] = torch.stack(
        [2 * a**2 - 1 + 2 * b**2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c]
    ).T
    rotation[:, 1, :] = torch.stack(
        [2 * b * c + 2 * a * d, 2 * a**2 - 1 + 2 * c**2, 2 * c * d - 2 * a * b]
    ).T
    rotation[:, 2, :] = torch.stack(
        [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, 2 * a**2 - 1 + 2 * d**2]
    ).T
    return rotation


def dummy_fill(
    coords_in: TensorType["n a 3"], atom_mask: TensorType["n a"], mode: str
) -> TensorType["n a 3"]:
    """
    Fill in ghost side chain atoms with either the CA or CB atom value for each residue, depending on the mode.

    Default to zeros if mode is not "CA" or "CB"
    """
    assert mode in ["CA", "CB"], f"Invalid dummy fill mode: {mode}"
    dummy_fill_mask = 1 - atom_mask
    if mode == "CA":
        dummy_fill_value = coords_in[..., 1:2, :]  # CA
    elif mode == "CB":
        dummy_fill_value = atom.fill_in_cbeta_for_atom37(coords_in)[
            ..., 3:4, :
        ]  # idealized CB
    else:
        dummy_fill_value = 0
    coords_in = (
        coords_in * atom_mask.unsqueeze(-1) + dummy_fill_value * dummy_fill_mask.unsqueeze(-1)
    )
    return coords_in


def get_masked_coords_array(coords, atom_mask):
    ma_mask = repeat(1 - atom_mask.unsqueeze(-1).cpu().numpy(), "... 1 -> ... 3")
    return np.ma.array(coords.cpu().numpy(), mask=ma_mask)


def make_crop_cond_mask_and_recenter_coords(
    atom_mask: TensorType["b n a", float],
    atom_coords: TensorType["b n a c", float],
    aatype: TensorType["b n", int] | None = None,
    chain_index: TensorType["b n", int] | None = None,
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
            units as `atom_coords`). Defaults to 8.0.
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
    n_res = seq_mask.sum(-1)
    masks = []
    all_hotspot_masks = torch.zeros((b, n), device=device)

    for i, nr in enumerate(n_res):
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
            n_neighbors = close_mask.sum().int()

            # pick how many neighbors (up to 10)
            n_sele = torch.randint(
                2,
                n_neighbors.clamp(min=3, max=max_discontiguous_res + 1),
                (1,),
                device=device,
            )

            # Select the indices of CB atoms that are close together
            idxs = torch.arange(n, device=device)[close_mask.bool()]
            idxs = idxs[torch.randperm(len(idxs))[:n_sele]]

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
                    motif_aatype_str = restype_1to3[order_restype[aatype_int.item()]]

                    # Original Protpardelle tip atom definition, also used in La-Proteina
                    tip_atomtypes = RFDIFFUSION_BENCHMARK_TIP_ATOMS[motif_aatype_str]
                    tip_atom_idx_atom37 = [
                        atom_order.get(atype) for atype in tip_atomtypes
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
        overfit: int = -1,
        short_epoch: bool = False,
        se3_data_augment: bool = True,
        translation_scale: float = 1.0,
        chain_residx_gap: int = 200,
        dummy_fill_mode: str = "zero",
        subset: str | float = "",
    ):
        """_summary_

        Parameters
        ----------
        pdb_path : str
            Path to PDB file
        fixed_size : int
            Length dimensionality to trim/pad tensors.
        mode : str, optional
            Training or evaluation mode, by default "train"
        overfit : int, optional
            For debugging: if > 0, overfits to a small subset, by default -1
        short_epoch : bool, optional
            For debugging: if True, stop an epoch early , by default False
        se3_data_augment : bool, optional
            Randomly rotate and translate the input coordinates, by default True
        translation_scale : float, optional
            Gaussian noise to add as translation perturbation, by default 1.0
        chain_residx_gap : int, optional
            Amount to add to the residue index that spaces two chains, by default 200
        dummy_fill_mode : str, optional
            How to fill the non-existing atom coordinate dimensions, by default "zero"
        subset : str | float, optional
            Dataset-specific subset tag to train on, by default ""
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
                print(
                    f"Filtering by resolution better than {subset}A and total length less than {self.fixed_size} residues. Removed {orig_size - new_size} examples, will train on {new_size} examples."
                )
            self.pdb_keys = list(
                set(
                    self.cif_df["chain_1_cluster_id"].to_list()
                    + self.cif_df["chain_2_cluster_id"].to_list()
                )
            )
        else:
            with open(f"{self.pdb_path}/{mode}_{subset}pdb_keys.list", "r") as f:
                self.pdb_keys = np.array(f.read().split("\n")[:-1])

        if overfit > 0:
            n_data = len(self.pdb_keys)
            self.pdb_keys = np.random.choice(
                self.pdb_keys, min(n_data, overfit), replace=False
            ).repeat(n_data // overfit)

    def __len__(self):
        if self.short_epoch:
            return min(len(self.pdb_keys), 256)
        else:
            return len(self.pdb_keys)

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
            example, _ = protpardelle.data.pdb_io.load_feats_from_pdb(
                data_file, chain_residx_gap=self.chain_residx_gap, chain_id=chain_id
            )
            coords_in = example["atom_positions"]
        except Exception as e:
            print(f"File {data_file} throws {e}")
            return

        # Apply data augmentation
        if self.se3_data_augment:
            coords_in, random_rot = apply_random_se3(
                coords_in,
                atom_mask=example["atom_mask"],
                translation_scale=self.translation_scale,
                return_rot=True,
            )
        else:
            coords_mean = coords_in[:, 1:2].mean(-3, keepdim=True)
            coords_in -= coords_mean

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
            coords_in = dummy_fill(
                coords_in, example["atom_mask"], mode=self.dummy_fill_mode
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
                fixed_size_example[k] = make_fixed_size_2d(
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
    """
    Stochastic Mixed Sampler.

    A sampler to draw samples from multiple datasets. This sampler is specifically designed
    to accommodate a setup where there's one primary dataset that needs to be fully iterated
    without replacement (typically primary examples) and several augmented datasets where samples
    can be drawn with replacement. The ratio of drawing from each dataset is determined by the
    provided mixing ratios, but we are guaranteed to draw a fixed number of samples from the
    primary dataset (equal to int(batch_size * mixing_ratios[0])).

    Parameters:
    - datasets (list[Dataset]): A list of datasets to sample from. The first dataset is considered
                                the primary dataset that will be sampled without replacement.
    - mixing_ratios (list[float]): A list of floats representing the mixing ratio for each dataset.
                                   The sum of all ratios should be equal to 1. The length of this list
                                   should be equal to the number of datasets.
    - batch_size (int): The batch size for which samples need to be drawn.

    Attributes:
    - primary_dataset_length (int): The length of the primary dataset.
    - primary_samples_per_batch (int): The number of primary dataset examples we draw per batch.
    - offsets (list[int]): The accumulated offset of each dataset when all datasets are concatenated.
    - samplers (list[Sampler]): List of samplers for each dataset. The primary dataset uses a
                                RandomSampler without replacement while all others use a RandomSampler with replacement.

    Methods:
    - __iter__: Returns an iterator that provides indices to draw samples from the combined dataset.
    - __len__: Returns the approximate length of indices that will be drawn in total.
    """

    def __init__(
        self, datasets: list[Dataset], mixing_ratios: list[float], batch_size: int
    ):
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

    def __len__(self):
        return (
            int(np.ceil(len(self.datasets[0]) / self.primary_samples_per_batch))
            * self.batch_size
        )


def calc_sigma_data(
    dataset: PDBDataset,
    config: argparse.Namespace,
    num_workers: int,
) -> float:
    """
    Given a dataset and the model training config, estimate sigma_data.
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
            coords, _, _ = make_crop_cond_mask_and_recenter_coords(
                atom_mask, coords, **vars(config.train.crop_cond)
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
        atom_mask = torch.ones_like(
            atom_mask
        ) * protpardelle.utils.unsqueeze_trailing_dims(seq_mask, atom_mask)

    # Estimate sigma_data
    masked_coords = get_masked_coords_array(coords, atom_mask)
    sigma_data = round(float(masked_coords.std()), 2)
    print(f"mean={masked_coords.mean():.2f}, sigma_data={sigma_data}")
    return sigma_data
