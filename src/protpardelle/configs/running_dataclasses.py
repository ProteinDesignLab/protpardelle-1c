"""Data classes for running configuration.

Author: Zhaoyang Li
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class NoiseSchedule:
    _partial_: bool
    _target_: str
    s_max: float
    s_min: float


@dataclass
class PartialDiffusion:
    enabled: bool
    pdb_file_path: str | None
    num_steps: int
    repack: bool | None
    seq: str | None


@dataclass
class DiscontiguousMotifAssignment:
    enabled: bool
    strategy: Literal["fixed"]
    # strategy: str
    fixed_motif_pos: list[int]


@dataclass
class CropConditionalGuidance:
    enabled: bool
    start: float
    end: float
    freq: int
    freq_start: float
    freq_end: float
    strategy: Literal["backbone", "sidechain", "sidechain-tip", "backbone-sidechain"]
    # strategy: str


@dataclass
class LossWeights:
    motif: float


@dataclass
class ReconstructionGuidance:
    enabled: bool
    start: float
    end: float
    schedule: Literal["constant", "quadratic", "cubic", "custom"]
    # schedule: str
    max_scale: float
    loss_weights: LossWeights


@dataclass
class ReplacementGuidance:
    enabled: bool
    start: float
    end: float


@dataclass
class ConditionalCfg:
    enabled: bool
    discontiguous_motif_assignment: DiscontiguousMotifAssignment
    num_recurrence_steps: int
    crop_conditional_guidance: CropConditionalGuidance
    reconstruction_guidance: ReconstructionGuidance
    replacement_guidance: ReplacementGuidance


@dataclass
class AllAtomCfg:
    sidechain_mode: bool
    skip_mpnn_proportion: float
    use_fullmpnn: bool
    use_fullmpnn_for_final: bool
    anneal_seq_resampling_rate: Literal["linear"]
    # anneal_seq_resampling_rate: str
    jump_steps: bool
    uniform_steps: bool


@dataclass
class Stage2Cfg:
    enabled: bool
    rewind_steps: int
    num_steps: int
    s_churn: int
    step_scale: float
    sidechain_mode: bool
    skip_mpnn_proportion: float
    noise_schedule: NoiseSchedule


@dataclass
class Sampling:
    num_steps: int
    s_churn: int
    step_scale: float
    noise_schedule: NoiseSchedule

    motif_file_path: str | None
    partial_diffusion: PartialDiffusion

    dx: float
    dy: float
    dz: float

    conditional_cfg: ConditionalCfg
    allatom_cfg: AllAtomCfg
    stage2_cfg: Stage2Cfg


@dataclass
class Run:
    dir: str


@dataclass
class Hydra:
    run: Run


@dataclass
class RunningConfig:
    sampling: Sampling
    hydra: Hydra
