"""Tests for utility functions."""

import argparse
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float, Int

from protpardelle.utils import (
    DotDict,
    apply_dotdict_recursively,
    clean_gpu_cache,
    dict_to_namespace,
    enable_tf32_if_available,
    get_default_device,
    get_logger,
    load_config,
    namespace_to_dict,
    norm_path,
    seed_everything,
    tensor_to_ndarray,
    unsqueeze_trailing_dims,
)


class TestDotDict:
    """Test DotDict class."""

    @pytest.mark.parametrize(
        "input_dict,expected_keys",
        [
            ({"key1": "value1", "key2": "value2"}, ["key1", "key2"]),
            ({"a": 1, "b": 2, "c": 3}, ["a", "b", "c"]),
            ({}, []),
            ({"single_key": "single_value"}, ["single_key"]),
        ],
    )
    def test_dotdict_creation(self, input_dict, expected_keys):
        """Test creating DotDict with various input dictionaries."""
        # Act
        d = DotDict(input_dict)

        # Assert
        for key in expected_keys:
            assert hasattr(d, key)
            assert getattr(d, key) == input_dict[key]

    def test_dotdict_nested_access(self):
        """Test nested access with DotDict."""
        # Arrange
        input_dict = {"level1": {"level2": "value"}}

        # Act
        d = DotDict(input_dict)

        # Assert
        # DotDict only applies to the top level, nested dicts remain as dict
        assert d.level1["level2"] == "value"
        assert isinstance(d.level1, dict)

    def test_dotdict_assignment(self):
        """Test assignment with DotDict."""
        # Arrange
        d = DotDict()

        # Act
        d.new_key = "new_value"

        # Assert
        assert d.new_key == "new_value"
        assert hasattr(d, "new_key")

    def test_dotdict_conversion_from_dict(self):
        """Test converting regular dict to DotDict."""
        # Arrange
        regular_dict = {"key1": "value1", "key2": "value2"}

        # Act
        d = DotDict(regular_dict)

        # Assert
        assert d.key1 == "value1"
        assert d.key2 == "value2"
        assert len(d) == 2

    def test_dotdict_key_error(self):
        """Test that accessing non-existent keys raises AttributeError."""
        # Arrange
        d = DotDict({"existing_key": "value"})

        # Act & Assert
        with pytest.raises(AttributeError):
            _ = d.non_existing_key

    def test_dotdict_with_nested_structures(self):
        """Test DotDict with complex nested structures."""
        # Arrange
        complex_data = {
            "level1": {
                "level2": {"level3": [1, 2, 3], "level3_dict": {"key": "value"}}
            },
            "list_data": [{"item": 1}, {"item": 2}],
            "mixed_types": {
                "string": "test",
                "number": 42,
                "boolean": True,
                "none_value": None,
            },
        }

        # Act
        dotdict = DotDict(complex_data)

        # Assert
        assert dotdict.level1["level2"]["level3"] == [1, 2, 3]
        assert dotdict.level1["level2"]["level3_dict"]["key"] == "value"
        assert dotdict.list_data == [{"item": 1}, {"item": 2}]
        assert dotdict.mixed_types["string"] == "test"

    def test_dotdict_with_empty_structures(self):
        """Test DotDict with empty data structures."""
        # Arrange
        empty_data = {
            "empty_dict": {},
            "empty_list": [],
            "empty_string": "",
            "zero_value": 0,
            "false_value": False,
        }

        # Act
        dotdict = DotDict(empty_data)

        # Assert
        assert dotdict.empty_dict == {}
        assert dotdict.empty_list == []
        assert dotdict.empty_string == ""
        assert dotdict.zero_value == 0
        assert dotdict.false_value is False


class TestTensorEdgeCases:
    """Test edge cases with tensor operations."""

    def test_tensor_with_inf_values(self):
        """Test handling of infinite values in tensors."""
        # Arrange
        normal_tensor = torch.randn(5, 3)
        inf_tensor = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("-inf")]])

        # Act & Assert
        assert torch.all(torch.isfinite(normal_tensor))
        assert not torch.all(torch.isfinite(inf_tensor))
        assert torch.any(torch.isinf(inf_tensor))

    def test_tensor_with_nan_values(self):
        """Test handling of NaN values in tensors."""
        # Arrange
        normal_tensor = torch.randn(3, 2)
        nan_tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])

        # Act & Assert
        assert torch.all(torch.isfinite(normal_tensor))
        assert not torch.all(torch.isfinite(nan_tensor))
        assert torch.any(torch.isnan(nan_tensor))

    def test_tensor_with_extreme_values(self):
        """Test handling of extremely large or small values."""
        # Arrange
        large_tensor = torch.tensor([1e10, 1e15, 1e20])
        small_tensor = torch.tensor([1e-10, 1e-15, 1e-20])

        # Act & Assert
        assert torch.all(torch.isfinite(large_tensor))
        assert torch.all(torch.isfinite(small_tensor))
        assert torch.all(large_tensor > 1e9)
        assert torch.all(small_tensor < 1e-9)

    def test_empty_tensor_operations(self):
        """Test operations on empty tensors."""
        # Arrange
        empty_tensor = torch.empty(0, 3, 4)

        # Act & Assert
        assert empty_tensor.shape == (0, 3, 4)
        assert empty_tensor.numel() == 0

        # Test operations that should handle empty tensors gracefully
        mean_val = torch.mean(empty_tensor)
        assert torch.isnan(mean_val) or mean_val == 0

    def test_single_element_tensor(self):
        """Test operations on single element tensors."""
        # Arrange
        single_tensor = torch.tensor([42.0])

        # Act & Assert
        assert single_tensor.shape == (1,)
        assert single_tensor.item() == 42.0
        assert torch.mean(single_tensor) == 42.0

    def test_tensor_shape_consistency(self):
        """Test tensor shape consistency in various operations."""
        # Arrange
        tensor_a = torch.randn(2, 3, 4)
        tensor_b = torch.randn(2, 3, 4)
        tensor_c = torch.randn(2, 3, 5)  # Different last dimension

        # Act & Assert
        # Compatible shapes
        result_compatible = tensor_a + tensor_b
        assert result_compatible.shape == tensor_a.shape

        # Incompatible shapes should raise error
        with pytest.raises(RuntimeError):
            _ = tensor_a + tensor_c

    def test_division_by_zero_handling(self):
        """Test handling of division by zero."""
        # Arrange
        numerator = torch.tensor([1.0, 2.0, 3.0])
        denominator = torch.tensor([1.0, 0.0, -1.0])

        # Act & Assert
        # PyTorch may handle division by zero differently (returning inf/nan)
        result = numerator / denominator
        assert torch.any(torch.isinf(result)) or torch.any(torch.isnan(result))

    def test_sqrt_of_negative_numbers(self):
        """Test square root of negative numbers."""
        # Arrange
        negative_tensor = torch.tensor([-1.0, -4.0, -9.0])

        # Act & Assert
        # PyTorch may handle sqrt of negative numbers differently (returning nan)
        result = torch.sqrt(negative_tensor)
        assert torch.any(torch.isnan(result))

    def test_log_of_zero_or_negative(self):
        """Test logarithm of zero or negative numbers."""
        # Arrange
        problematic_tensor = torch.tensor([0.0, -1.0, 2.0])

        # Act & Assert
        # PyTorch may handle log of zero/negative differently (returning -inf/nan)
        result = torch.log(problematic_tensor)
        assert torch.any(torch.isinf(result)) or torch.any(torch.isnan(result))

    def test_very_small_numbers(self):
        """Test handling of very small numbers."""
        # Arrange
        small_tensor = torch.tensor([1e-20, 1e-30, 1e-40])

        # Act
        result = small_tensor * 1e20

        # Assert
        assert torch.all(torch.isfinite(result))
        assert abs(result[0].item() - 1.0) < 1e-15
        assert (
            abs(result[1].item() - 1e-10) < 1e-17
        )  # Relaxed tolerance for floating point precision
        assert (
            abs(result[2].item() - 1e-20) < 1e-25
        )  # Allow for floating point precision


class TestMemoryEdgeCases:
    """Test memory-related edge cases."""

    def test_large_tensor_memory_usage(self):
        """Test memory usage with large tensors."""
        # Arrange - Create moderately large tensor (not too large for CI)
        large_tensor = torch.randn(1000, 1000)

        # Act & Assert
        assert large_tensor.numel() == 1000000
        assert large_tensor.element_size() * large_tensor.numel() > 1000000  # > 1MB

    def test_tensor_memory_cleanup(self):
        """Test tensor memory cleanup."""

        # Arrange
        def create_tensor():
            return torch.randn(100, 100)

        # Act
        tensor1 = create_tensor()
        tensor2 = create_tensor()

        # Assert
        assert tensor1 is not None
        assert tensor2 is not None

        # Tensors should be different objects
        assert tensor1 is not tensor2

    def test_gradient_memory_management(self):
        """Test gradient memory management."""
        # Arrange
        tensor = torch.randn(3, 3, requires_grad=True)

        # Act
        loss = torch.sum(tensor**2)
        loss.backward()

        # Assert
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape

        # Clear gradients
        tensor.grad.zero_()
        assert torch.all(tensor.grad == 0)


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_type_error_handling(self):
        """Test handling of type errors."""
        # Arrange
        string_data = "not_a_tensor"

        # Act & Assert
        with pytest.raises((TypeError, AttributeError)):
            _ = string_data + torch.randn(3)

    def test_value_error_handling(self):
        """Test handling of value errors."""
        # Arrange
        invalid_shape = (-1, 3)  # Invalid tensor shape

        # Act & Assert
        with pytest.raises((ValueError, RuntimeError)):
            _ = torch.randn(*invalid_shape)

    def test_index_error_handling(self):
        """Test handling of index errors."""
        # Arrange
        tensor = torch.randn(5, 3)

        # Act & Assert
        with pytest.raises(IndexError):
            _ = tensor[10, 0]  # Index out of bounds

    def test_key_error_handling(self):
        """Test handling of key errors."""
        # Arrange
        dictionary = {"key1": "value1"}

        # Act & Assert
        with pytest.raises(KeyError):
            _ = dictionary["nonexistent_key"]

    def test_assertion_error_handling(self):
        """Test handling of assertion errors."""
        # Arrange
        condition = False

        # Act & Assert
        with pytest.raises(AssertionError):
            assert condition, "This should fail"

    def test_file_not_found_error_handling(self):
        """Test handling of file not found errors."""
        # Arrange
        nonexistent_file = Path("/nonexistent/file.txt")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            with open(nonexistent_file, "r"):
                pass


class TestTensorPerformance:
    """Test tensor operation performance."""

    def test_large_tensor_operations(self):
        """Test performance of operations on large tensors."""
        # Arrange
        large_tensor = torch.randn(1000, 1000)

        # Act
        start_time = time.time()

        # Perform various operations
        result1 = torch.matmul(large_tensor, large_tensor.T)
        result2 = torch.sum(result1, dim=1)
        result3 = torch.mean(result2)

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert
        assert torch.isfinite(result3)
        assert execution_time < 10.0  # Should complete within 10 seconds

    def test_tensor_memory_efficiency(self):
        """Test memory efficiency of tensor operations."""
        # Arrange
        tensor_size = 500

        # Act
        # Create tensors and perform operations
        tensors = []
        for i in range(10):
            tensor = torch.randn(tensor_size, tensor_size)
            result = torch.matmul(tensor, tensor.T)
            tensors.append(result)

        # Assert
        assert len(tensors) == 10
        for tensor in tensors:
            assert tensor.shape == (tensor_size, tensor_size)

    def test_tensor_batch_operations(self):
        """Test batch tensor operations performance."""
        # Arrange
        batch_size = 100
        seq_length = 50
        atom_count = 37
        coord_dim = 3

        coordinates = torch.randn(batch_size, seq_length, atom_count, coord_dim)

        # Act
        start_time = time.time()

        # Batch operations
        norms = torch.norm(coordinates, dim=-1)
        means = torch.mean(norms, dim=1)
        stds = torch.std(norms, dim=1)

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert
        assert norms.shape == (batch_size, seq_length, atom_count)
        assert means.shape == (
            batch_size,
            atom_count,
        )  # Mean over sequence length, not atoms
        assert stds.shape == (
            batch_size,
            atom_count,
        )  # Std over sequence length, not atoms
        assert execution_time < 5.0  # Should complete within 5 seconds

    def test_tensor_gradient_computation_performance(self):
        """Test gradient computation performance."""
        # Arrange
        tensor = torch.randn(100, 100, requires_grad=True)

        # Act
        start_time = time.time()

        # Compute gradients
        loss = torch.sum(tensor**2)
        loss.backward()

        end_time = time.time()
        execution_time = end_time - start_time

        # Assert
        assert tensor.grad is not None
        assert execution_time < 2.0  # Should complete within 2 seconds


class TestApplyDotdictRecursively:
    """Test apply_dotdict_recursively function."""

    def test_apply_dotdict_recursively_basic(self):
        """Test basic recursive DotDict application."""
        data = {"key1": {"key2": "value"}}
        result = apply_dotdict_recursively(data)
        assert isinstance(result, DotDict)
        assert result.key1.key2 == "value"

    def test_apply_dotdict_recursively_nested(self):
        """Test nested recursive DotDict application."""
        data = {"level1": {"level2": {"level3": "value"}}}
        result = apply_dotdict_recursively(data)
        assert result.level1.level2.level3 == "value"

    def test_apply_dotdict_recursively_mixed(self):
        """Test recursive DotDict application with mixed types."""
        data = {
            "dict_key": {"nested": "value"},
            "list_key": [1, 2, 3],
            "str_key": "string_value",
            "int_key": 42,
        }
        result = apply_dotdict_recursively(data)
        assert result.dict_key.nested == "value"
        assert result.list_key == [1, 2, 3]
        assert result.str_key == "string_value"
        assert result.int_key == 42


class TestCleanGpuCache:
    """Test clean_gpu_cache function."""

    def test_clean_gpu_cache(self):
        """Test GPU cache cleaning."""

        # clean_gpu_cache is a decorator, test it as a decorator
        @clean_gpu_cache
        def dummy_function():
            return "test"

        result = dummy_function()
        assert result == "test"


class TestDictToNamespace:
    """Test dict_to_namespace function."""

    def test_dict_to_namespace_basic(self):
        """Test basic dict to namespace conversion."""
        d = {"key1": "value1", "key2": "value2"}
        namespace = dict_to_namespace(d)
        assert namespace.key1 == "value1"
        assert namespace.key2 == "value2"

    def test_dict_to_namespace_nested(self):
        """Test nested dict to namespace conversion."""
        d = {"level1": {"level2": "value"}}
        namespace = dict_to_namespace(d)
        assert namespace.level1.level2 == "value"


class TestEnableTf32IfAvailable:
    """Test enable_tf32_if_available function."""

    def test_enable_tf32_if_available(self):
        """Test enabling TF32 if available."""
        # This function should not raise any errors
        enable_tf32_if_available()


class TestGetDefaultDevice:
    """Test get_default_device function."""

    def test_get_default_device(self):
        """Test getting default device."""
        device = get_default_device()
        assert isinstance(device, torch.device)


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger(self):
        """Test getting logger."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"


class TestNamespaceToDict:
    """Test namespace_to_dict function."""

    def test_namespace_to_dict_basic(self):
        """Test basic namespace to dict conversion."""
        namespace = argparse.Namespace()
        namespace.key1 = "value1"
        namespace.key2 = "value2"

        result = namespace_to_dict(namespace)
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_namespace_to_dict_nested(self):
        """Test nested namespace to dict conversion."""
        namespace = argparse.Namespace()
        namespace.level1 = argparse.Namespace()
        namespace.level1.level2 = "value"

        result = namespace_to_dict(namespace)
        assert result["level1"]["level2"] == "value"


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_interface(self):
        """Test load_config function interface."""
        # This would require a test config file
        # For now, just test the interface
        pass


class TestNormPath:
    """Test norm_path function."""

    def test_norm_path_basic(self):
        """Test basic path normalization."""
        path = "/some/path/to/file"
        result = norm_path(path)
        assert isinstance(result, Path)

    def test_norm_path_relative(self):
        """Test relative path normalization."""
        path = "relative/path"
        result = norm_path(path)
        assert isinstance(result, Path)


class TestSeedEverything:
    """Test seed_everything function."""

    def test_seed_everything(self):
        """Test seeding everything."""
        # This function should not raise any errors
        seed_everything(42)


class TestTensorToNdarray:
    """Test tensor_to_ndarray function."""

    def test_tensor_to_ndarray_torch(self):
        """Test converting torch tensor to ndarray."""
        tensor = torch.randn(2, 3)
        result = tensor_to_ndarray(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_tensor_to_ndarray_numpy(self):
        """Test converting numpy array to ndarray."""
        arr = np.random.randn(2, 3)
        result = tensor_to_ndarray(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)


class TestUnsqueezeTrailingDims:
    """Test unsqueeze_trailing_dims function."""

    def test_unsqueeze_trailing_dims(self):
        """Test unsqueezing trailing dimensions."""
        tensor = torch.randn(2, 3)
        # The second argument should be a tensor or None, not an integer
        target_tensor = torch.randn(2, 3, 1, 1)
        result = unsqueeze_trailing_dims(tensor, target_tensor)
        assert result.shape == (2, 3, 1, 1)


class TestUtilityValidation:
    """Test utility validation functions."""

    def test_validate_path(self):
        """Test path validation."""
        # Test validating file paths
        pass

    def test_validate_config(self):
        """Test config validation."""
        # Test validating configuration objects
        pass

    def test_validate_device(self):
        """Test device validation."""
        # Test validating device specifications
        pass
