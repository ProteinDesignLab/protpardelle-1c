# ProtPardelle Test Suite

This directory contains comprehensive tests for the ProtPardelle protein generation framework. The test suite is organized using modern Python testing practices with pytest.

## Test Organization

The test suite follows a modular structure that mirrors the source code organization:

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── pytest.ini                 # Pytest configuration
├── README.md                  # This file
│
├── test_common/               # Tests for common modules
│   ├── test_common.py         # Common utilities and protein dataclasses
│   ├── test_protein.py        # Protein structure tests
│   └── test_residue_constants.py # Residue constants tests
│
├── test_core/                 # Tests for core functionality
│   ├── test_diffusion.py      # Diffusion utilities
│   ├── test_modules.py        # Core neural network modules
│   ├── test_models.py         # Model definitions and utilities
│   ├── test_core_models.py    # Core model tests
│   └── test_cyclic_peptides.py # Cyclic peptide utilities
│
├── test_data/                 # Tests for data processing
│   ├── test_atom.py          # Atom coordinate utilities
│   ├── test_pdb_io.py        # PDB file I/O
│   ├── test_sequence.py      # Sequence processing
│   ├── test_dataset.py       # Dataset utilities
│   ├── test_data.py          # Data processing utilities
│   ├── test_motif_placement.py # Motif placement utilities
│   ├── test_align.py         # Alignment utilities
│   ├── test_cycpep.py        # Cyclic peptide processing
│   └── test_motif.py          # Motif utilities
│
├── test_integrations/         # Tests for integrations
│   └── test_integrations.py   # ESMFold and ProteinMPNN integration tests
│
├── test_configs/             # Tests for configuration
│   ├── test_configs.py       # Configuration tests
│   └── test_dataclasses.py   # Configuration dataclasses
│
├── test_env.py               # Environment utilities tests
├── test_evaluate.py          # Evaluation utilities tests
├── test_likelihood.py        # Likelihood computation tests
├── test_sample.py            # Sampling functionality tests
├── test_train.py             # Training functionality tests
├── test_utils.py             # Utility function tests
└── test_sample_train.py      # Combined sampling and training tests
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run tests in a specific module
pytest tests/test_core/

# Run tests with verbose output
pytest -v

# Run tests and show coverage
pytest --cov=src/protpardelle --cov-report=html
```

### Test Categories

Tests are organized by category using pytest markers:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests (if no GPU available)
pytest -m "not gpu"

# Skip tests requiring models/checkpoints
pytest -m "not requires_model"
```

### Parallel Execution

For faster test execution, you can run tests in parallel:

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Test Categories

### Unit Tests

- **Core modules**: Neural network components, diffusion utilities
- **Data processing**: PDB I/O, sequence processing, atom utilities
- **Configuration**: Dataclass validation and parsing
- **Utilities**: Helper functions and common utilities

### Integration Tests

- **Sampling workflows**: End-to-end protein generation
- **Model loading**: Loading and inference with trained models
- **Data pipelines**: Complete data processing workflows

### Performance Tests

- **Memory usage**: Large structure handling
- **GPU utilization**: CUDA operations and memory management
- **Batch processing**: Efficient batch operations

## Fixtures and Test Utilities

The test suite includes comprehensive fixtures in `conftest.py`:

- **Device fixtures**: CPU and GPU device management
- **Sample data**: Protein structures, sequences, coordinates
- **Temporary files**: PDB files, configuration files
- **Mock configurations**: Test configuration objects
- **Random seeds**: Reproducible test execution

## Writing New Tests

### Test Structure

Follow the existing patterns:

```python
class TestFeatureName:
    """Test the FeatureName class/function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = feature_function(input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
    
    @pytest.mark.parametrize("param1,param2", [
        (value1a, value2a),
        (value1b, value2b),
    ])
    def test_with_parameters(self, param1, param2):
        """Test with different parameters."""
        result = feature_function(param1, param2)
        assert result is not None
```

### Test Markers

Use appropriate markers for test categorization:

```python
@pytest.mark.slow
def test_expensive_operation():
    """Test that takes a long time to run."""
    pass

@pytest.mark.gpu
def test_gpu_functionality():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete workflow."""
    pass

@pytest.mark.requires_model
def test_with_trained_model():
    """Test that requires a trained model."""
    pass
```

### Fixtures

Use existing fixtures when possible:

```python
def test_with_sample_data(sample_coords, sample_aatype, device):
    """Test using sample data fixtures."""
    # Use the provided fixtures
    result = process_coordinates(sample_coords.to(device), sample_aatype)
    assert result is not None
```

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

- **GitHub Actions**: Configured to run on multiple Python versions
- **Coverage reporting**: HTML and terminal coverage reports
- **Test results**: JUnit XML output for CI integration
- **Parallel execution**: Optimized for CI environments

## Debugging Tests

### Verbose Output

```bash
# Maximum verbosity
pytest -vvv

# Show local variables in tracebacks
pytest --tb=long

# Drop into debugger on failures
pytest --pdb
```

### Test Selection

```bash
# Run specific test
pytest tests/test_core/test_modules.py::TestBuildCyclicHarmonics::test_basic_functionality

# Run tests matching pattern
pytest -k "test_cyclic"

# Run tests in specific file
pytest tests/test_data/test_atom.py
```

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=src/protpardelle --cov-report=html

# Show missing lines
pytest --cov=src/protpardelle --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=src/protpardelle --cov-fail-under=80
```

## Performance Considerations

### Test Optimization

- **Fast tests first**: Unit tests run before integration tests
- **Parallel execution**: Use `pytest-xdist` for parallel test execution
- **Selective testing**: Use markers to run only relevant tests during development
- **Mocking**: Use mocks for expensive operations in unit tests

### Memory Management

- **GPU memory**: Tests clean up GPU memory after execution
- **Large structures**: Use fixtures to manage large test data
- **Batch sizes**: Use appropriate batch sizes for memory-constrained environments

## Contributing

When adding new tests:

1. **Follow existing patterns**: Use the same structure and naming conventions
2. **Add appropriate markers**: Categorize tests with markers
3. **Include docstrings**: Document what each test verifies
4. **Use fixtures**: Leverage existing fixtures when possible
5. **Test edge cases**: Include tests for error conditions and edge cases
6. **Update documentation**: Update this README if adding new test categories

## Troubleshooting

### Common Issues

1. **GPU tests failing**: Ensure CUDA is available and properly configured
2. **Slow test execution**: Use markers to skip slow tests during development
3. **Memory issues**: Reduce batch sizes or use smaller test data
4. **Import errors**: Ensure the source package is properly installed

### Getting Help

- Check existing tests for examples
- Review pytest documentation
- Ask in project discussions or issues
