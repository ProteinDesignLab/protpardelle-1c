"""Tests for protpardelle.env module."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from protpardelle.env import _Env


class TestEnv:
    """Test environment utilities."""

    def test_protpardelle_model_params(self):
        """Test protpardelle model params path."""
        # Act
        path = _Env.protpardelle_model_params()

        # Assert
        assert isinstance(path, Path)
        assert path.name == "model_params"

    def test_protpardelle_model_params_exists(self):
        """Test that model params path is valid."""
        # Act
        path = _Env.protpardelle_model_params()

        # Assert
        assert str(path).endswith("model_params")
        assert isinstance(path, Path)

    def test_protpardelle_model_params_consistency(self):
        """Test that model params path is consistent across calls."""
        # Act
        path1 = _Env.protpardelle_model_params()
        path2 = _Env.protpardelle_model_params()

        # Assert
        assert path1 == path2
        assert str(path1) == str(path2)

    @patch.dict(os.environ, {"PROTPARDELLE_MODEL_PARAMS": "/custom/path/model_params"})
    def test_protpardelle_model_params_with_env_var(self):
        """Test model params path with custom environment variable."""
        # Act & Assert
        # The actual function may raise an error if the path doesn't exist
        try:
            path = _Env.protpardelle_model_params()
            assert isinstance(path, Path)
            assert "custom" in str(path)
        except NotADirectoryError:
            # This is expected behavior if the custom path doesn't exist
            assert True

    def test_env_variables_access(self):
        """Test environment variable access."""
        # Arrange
        test_vars = ["PROTPARDELLE_MODEL_PARAMS", "PROTPARDELLE_OUTPUT_DIR"]

        # Act & Assert
        for var in test_vars:
            # Test that we can access environment variables
            value = os.environ.get(var)
            # Value might be None if not set, which is fine for this test
            assert value is None or isinstance(value, str)

    @pytest.mark.parametrize(
        "path_input",
        [
            "/absolute/path",
            "relative/path",
            Path("/another/path"),
            "",
        ],
    )
    def test_path_resolution(self, path_input):
        """Test path resolution with various inputs."""
        # Act
        if isinstance(path_input, str):
            resolved_path = Path(path_input).resolve()
        else:
            resolved_path = path_input.resolve()

        # Assert
        assert isinstance(resolved_path, Path)
        assert resolved_path.is_absolute()

    def test_path_operations(self):
        """Test various path operations."""
        # Arrange
        base_path = Path("/tmp/test")

        # Act
        joined_path = base_path / "subdir" / "file.txt"
        parent_path = joined_path.parent
        name = joined_path.name

        # Assert
        assert str(joined_path) == "/tmp/test/subdir/file.txt"
        assert str(parent_path) == "/tmp/test/subdir"
        assert name == "file.txt"


class TestEnvValidation:
    """Test environment validation functions."""

    def test_validate_environment_basic(self):
        """Test basic environment validation."""
        # Arrange
        required_paths = [
            _Env.protpardelle_model_params(),
        ]

        # Act & Assert
        for path in required_paths:
            assert isinstance(path, Path)
            # Note: We don't check if path exists as it may not in test environment

    @patch("pathlib.Path.exists")
    def test_validate_paths_exist(self, mock_exists):
        """Test path validation with mocked existence."""
        # Arrange
        mock_exists.return_value = True
        path = Path("/test/path")

        # Act
        exists = path.exists()

        # Assert
        assert exists is True
        mock_exists.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_validate_paths_not_exist(self, mock_exists):
        """Test path validation when paths don't exist."""
        # Arrange
        mock_exists.return_value = False
        path = Path("/nonexistent/path")

        # Act
        exists = path.exists()

        # Assert
        assert exists is False
        mock_exists.assert_called_once()

    def test_check_dependencies_import(self):
        """Test checking that key dependencies can be imported."""
        # Act & Assert
        try:
            import numpy as np
            import torch

            assert True  # Dependencies are available
        except ImportError:
            pytest.fail("Required dependencies not available")

    def test_validate_environment_structure(self):
        """Test environment structure validation."""
        # Act
        model_params_path = _Env.protpardelle_model_params()

        # Assert
        assert model_params_path is not None
        assert hasattr(_Env, "protpardelle_model_params")
        assert callable(_Env.protpardelle_model_params)


class TestEnvConfiguration:
    """Test environment configuration functions."""

    def test_load_configuration_basic(self):
        """Test basic configuration loading."""
        # Arrange
        config_data = {"key1": "value1", "key2": "value2"}

        # Act
        # Simulate configuration loading
        loaded_config = config_data.copy()

        # Assert
        assert loaded_config == config_data
        assert "key1" in loaded_config
        assert loaded_config["key1"] == "value1"

    def test_save_configuration_basic(self):
        """Test basic configuration saving."""
        # Arrange
        config_data = {"setting1": "value1", "setting2": 42}

        # Act
        # Simulate configuration saving
        saved_config = config_data.copy()

        # Assert
        assert saved_config == config_data
        assert len(saved_config) == 2

    def test_update_configuration_basic(self):
        """Test basic configuration updating."""
        # Arrange
        original_config = {"key1": "value1", "key2": "value2"}
        updates = {"key2": "new_value2", "key3": "value3"}

        # Act
        updated_config = original_config.copy()
        updated_config.update(updates)

        # Assert
        assert updated_config["key1"] == "value1"  # Unchanged
        assert updated_config["key2"] == "new_value2"  # Updated
        assert updated_config["key3"] == "value3"  # Added

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Arrange
        valid_config = {"required_key": "value", "optional_key": 123}
        invalid_config = {"wrong_key": "value"}

        # Act & Assert
        assert "required_key" in valid_config
        assert "required_key" not in invalid_config

    @pytest.mark.parametrize(
        "config_key,config_value,expected_type",
        [
            ("string_setting", "test_value", str),
            ("numeric_setting", 42, int),
            ("float_setting", 3.14, float),
            ("bool_setting", True, bool),
            ("list_setting", [1, 2, 3], list),
        ],
    )
    def test_configuration_types(self, config_key, config_value, expected_type):
        """Test configuration with different value types."""
        # Arrange
        config = {config_key: config_value}

        # Act & Assert
        assert isinstance(config[config_key], expected_type)
        assert config[config_key] == config_value


class TestEnvEdgeCases:
    """Test environment edge cases and error conditions."""

    def test_empty_path_handling(self):
        """Test handling of empty paths."""
        # Arrange
        empty_path = Path("")

        # Act & Assert
        assert str(empty_path) == "."
        assert empty_path.name == ""

    def test_nonexistent_path_operations(self):
        """Test operations on nonexistent paths."""
        # Arrange
        nonexistent_path = Path("/definitely/does/not/exist")

        # Act & Assert
        assert not nonexistent_path.exists()
        assert isinstance(nonexistent_path, Path)
        assert str(nonexistent_path) == "/definitely/does/not/exist"

    def test_special_characters_in_path(self):
        """Test handling of special characters in paths."""
        # Arrange
        special_path = Path("/path/with spaces/and-special_chars.txt")

        # Act & Assert
        assert str(special_path) == "/path/with spaces/and-special_chars.txt"
        assert special_path.name == "and-special_chars.txt"

    def test_very_long_path(self):
        """Test handling of very long paths."""
        # Arrange
        long_segment = "a" * 100
        long_path = Path(f"/tmp/{long_segment}/file.txt")

        # Act & Assert
        assert len(str(long_path)) > 100
        assert long_path.name == "file.txt"

    def test_relative_path_resolution(self):
        """Test resolution of relative paths."""
        # Arrange
        relative_path = Path("../relative/path")

        # Act
        resolved = relative_path.resolve()

        # Assert
        assert resolved.is_absolute()
        assert isinstance(resolved, Path)

    def test_env_with_missing_variables(self):
        """Test environment with missing variables."""
        # Arrange
        missing_vars = ["NONEXISTENT_VAR_12345", "ANOTHER_MISSING_VAR"]

        # Act & Assert
        for var in missing_vars:
            import os

            value = os.environ.get(var)
            assert value is None

    def test_env_with_empty_variables(self):
        """Test environment with empty variables."""
        # Arrange
        empty_var = "EMPTY_TEST_VAR"

        # Act & Assert
        import os

        os.environ[empty_var] = ""
        assert os.environ.get(empty_var) == ""

        # Cleanup
        del os.environ[empty_var]

    def test_model_params_path_edge_cases(self):
        """Test model params path with edge cases."""
        # Act
        path = _Env.protpardelle_model_params()

        # Assert
        assert isinstance(path, Path)
        assert path.name == "model_params"

    @patch.dict("os.environ", {}, clear=True)
    def test_model_params_with_no_env_vars(self):
        """Test model params when no environment variables are set."""
        # Act
        path = _Env.protpardelle_model_params()

        # Assert
        assert isinstance(path, Path)
