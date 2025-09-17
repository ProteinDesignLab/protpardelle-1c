"""Initialization file for configuration data classes.

Author: Zhaoyang Li
"""

from typing import TypeAlias

from protpardelle.configs.running_dataclasses import ModelConfig
from protpardelle.configs.sampling_dataclasses import SamplingConfig
from protpardelle.configs.training_dataclasses import TrainingConfig

Config: TypeAlias = TrainingConfig | ModelConfig | SamplingConfig

__all__ = ["Config", "ModelConfig", "SamplingConfig", "TrainingConfig"]
