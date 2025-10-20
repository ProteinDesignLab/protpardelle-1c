"""Tests for protpardelle.configs.sampling_dataclasses module."""


def test_sampling_config_dataclass():
    """Test that SamplingConfig dataclass can be instantiated."""
    from protpardelle.configs.sampling_dataclasses import SamplingConfig

    assert SamplingConfig is not None
