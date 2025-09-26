"""Data classes for sampling configuration.

Author: Zhaoyang Li
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PartialDiffusion:
    enabled: bool
    rewind_steps: list[int]


@dataclass
class SearchSpace:
    crop_cond_starts: list[float]
    models: list[list[str]]
    schurns: list[float]
    step_scales: list[float]
    translations: list[list[float]]


@dataclass
class SamplingConfig:
    hotspots: list[str | None]
    motif_contigs: list[str | None]
    motifs: list[str | None]
    partial_diffusion: PartialDiffusion
    search_space: SearchSpace
    ssadj: list[int | None]
    total_lengths: list[list[list[int]] | None]
    cyclic_chains: list[list[str] | None]
