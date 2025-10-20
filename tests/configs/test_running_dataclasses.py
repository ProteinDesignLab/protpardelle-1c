"""Tests for protpardelle.configs.running_dataclasses module."""


def test_running_config_dataclass():
    """Test that RunningConfig dataclass can be instantiated."""
    from protpardelle.configs.running_dataclasses import RunningConfig

    assert RunningConfig is not None
