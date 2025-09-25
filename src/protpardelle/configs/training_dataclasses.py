"""Data classes for training configuration.

Author: Zhaoyang Li
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CropCond:
    contiguous_prob: float
    discontiguous_prob: float
    sidechain_prob: float
    sidechain_only_prob: float
    max_span_len: int
    max_discontiguous_res: int
    dist_threshold: float
    recenter_coords: bool

    multichain_prob: float | None
    hotspot_min: int | None
    hotspot_max: int | None
    hotspot_dropout: float | None
    paratope_prob: float | None


@dataclass
class Train:
    seed: int | None
    ckpt_path: str | None

    batch_size: int
    max_epochs: int
    eval_freq: int
    checkpoint_freq: int
    checkpoints: list[str]

    lr: float
    warmup_steps: int
    decay_steps: int
    use_amp: bool
    clip_grad_norm: bool
    grad_clip_val: float
    weight_decay: float

    n_eval_samples: int
    sc_num_seqs: int
    self_cond_train_prob: float
    subsample_eval_set: float

    crop_conditional: bool
    crop_cond: CropCond


@dataclass
class Data:
    pdb_paths: list[str]
    subset: list[str | float]
    mixing_ratios: list[float]

    fixed_size: int
    n_aatype_tokens: int

    short_epoch: int
    num_workers: int

    se3_data_augment: bool
    translation_scale: float

    chain_residx_gap: int

    sigma_data: float
    auto_calc_sigma_data: bool
    n_examples_for_sigma_data: int

    dummy_fill_mode: Literal["zero", "CA"]
    # dummy_fill_mode: str


@dataclass
class Training:
    function: Literal["uniform", "lognormal", "mpnn", "constant"]
    # function: str
    psigma_mean: float
    psigma_std: float


@dataclass
class Sampling:
    function: Literal["uniform", "lognormal", "mpnn", "constant"]
    # function: str
    s_min: float
    s_max: float


@dataclass
class Diffusion:
    training: Training
    sampling: Sampling


@dataclass
class UViT:
    patch_size: int
    n_layers: int
    n_heads: int
    dim_head: int
    n_filt_per_layer: list[int]
    n_blocks_per_layer: int
    cat_pwd_to_conv: bool
    conv_skip_connection: bool
    position_embedding_type: Literal[
        "rotary",
        "rotary_relchain",
        "absolute",
        "absolute_residx",
        "relative",
        "relative_relchain",
        "none",
    ]
    # position_embedding_type: str
    position_embedding_max: int


@dataclass
class StructModel:
    arch: Literal["dit", "uvit"]
    # arch: str
    n_atoms: int
    n_channel: int
    noise_cond_mult: int
    uvit: UViT


@dataclass
class MPNNModel:
    use_self_conditioning: bool
    label_smoothing: float
    n_channel: int
    n_layers: int
    n_neighbors: int
    noise_cond_mult: int


@dataclass
class Model:
    task: Literal["backbone", "allatom", "seqdes", "codesign"]
    # task: str
    pretrained_modules: list[Literal["struct_model", "mpnn_model"]]
    # pretrained_modules: list[str]
    struct_model_checkpoint: str
    mpnn_model_checkpoint: str
    crop_conditional: bool
    conditioning_style: Literal[
        "concat", "noise_residual", "concat_and_noise_residual"
    ]  # this should be always present
    # conditioning_style: str
    compute_loss_on_all_atoms: bool
    struct_model: StructModel
    mpnn_model: MPNNModel


@dataclass
class TrainingConfig:
    train: Train
    data: Data
    diffusion: Diffusion
    model: Model
