"""Initialization file for configuration data classes.

Author: Zhaoyang Li
"""

from typing import TypeAlias

from protpardelle.configs.running_dataclasses import RunningConfig
from protpardelle.configs.sampling_dataclasses import SamplingConfig
from protpardelle.configs.training_dataclasses import TrainingConfig

Config: TypeAlias = RunningConfig | SamplingConfig | TrainingConfig

__all__ = ["Config", "RunningConfig", "SamplingConfig", "TrainingConfig"]
