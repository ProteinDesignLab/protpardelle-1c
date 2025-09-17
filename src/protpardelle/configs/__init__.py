from typing import TypeAlias

from protpardelle.configs.model_dataclasses import ModelConfig
from protpardelle.configs.sample_dataclasses import SamplingConfig
from protpardelle.configs.train_dataclasses import TrainingConfig

Config: TypeAlias = TrainingConfig | ModelConfig | SamplingConfig

__all__ = ["Config", "ModelConfig", "SamplingConfig", "TrainingConfig"]
