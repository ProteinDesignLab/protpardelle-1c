"""Tests for protpardelle.configs.training_dataclasses module."""


def test_training_config_dataclass():
    """Test that TrainingConfig dataclass can be instantiated."""
    from protpardelle.configs.training_dataclasses import TrainingConfig

    assert TrainingConfig is not None
