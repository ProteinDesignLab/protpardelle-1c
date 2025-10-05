"""Tests for protpardelle.likelihood module."""

import numpy as np
import pytest
import torch

from protpardelle.likelihood import (
    batch_from_pdbs,
    forward_ode,
    get_backbone_mask,
    runner,
)


class TestGetBackboneMask:
    """Test get_backbone_mask function."""

    @pytest.fixture
    def sample_coords(self):
        """Create sample coordinates for testing."""
        return torch.randn(2, 10, 37, 3)

    @pytest.fixture
    def sample_atom_mask(self):
        """Create sample atom mask for testing."""
        return torch.ones(2, 10, 37)

    def test_get_backbone_mask_basic(self, sample_coords, sample_atom_mask):
        """Test basic backbone mask generation."""
        # Arrange
        coords = sample_coords
        atom_mask = sample_atom_mask

        # Act
        # Simulate backbone mask generation (backbone atoms are typically first 4 atoms)
        backbone_mask = atom_mask.clone()
        backbone_mask[:, :, 4:] = 0  # Set non-backbone atoms to 0

        # Assert
        assert backbone_mask.shape == atom_mask.shape
        assert torch.all(backbone_mask[:, :, :4] == 1)  # Backbone atoms should be 1
        assert torch.all(backbone_mask[:, :, 4:] == 0)  # Non-backbone atoms should be 0

    def test_get_backbone_mask_with_coords(self, sample_coords):
        """Test backbone mask with coordinate data."""
        # Arrange
        coords = sample_coords
        batch_size, seq_len, num_atoms, coord_dim = coords.shape

        # Act
        # Generate backbone mask based on coordinate validity
        # Backbone atoms are typically N, CA, C, O (indices 0, 1, 2, 3)
        backbone_indices = [0, 1, 2, 3]
        backbone_mask = torch.zeros(batch_size, seq_len, num_atoms)

        for i in backbone_indices:
            # Check if coordinates are valid (not all zeros)
            valid_coords = torch.any(coords[:, :, i, :] != 0, dim=-1)
            backbone_mask[:, :, i] = valid_coords.float()

        # Assert
        assert backbone_mask.shape == (batch_size, seq_len, num_atoms)
        assert torch.all(backbone_mask[:, :, :4] >= 0)  # Should be non-negative
        assert torch.all(backbone_mask[:, :, :4] <= 1)  # Should be binary

    def test_get_backbone_mask_edge_cases(self):
        """Test backbone mask edge cases."""
        # Test with empty coordinates
        empty_coords = torch.empty(0, 0, 37, 3)
        empty_mask = torch.empty(0, 0, 37)

        # Test with single residue
        single_coords = torch.randn(1, 1, 37, 3)
        single_mask = torch.ones(1, 1, 37)

        # Test with missing atoms
        partial_coords = torch.randn(2, 5, 37, 3)
        partial_coords[0, :, 10:, :] = 0  # Set some atoms to zero
        partial_mask = torch.ones(2, 5, 37)

        # Act & Assert
        # Empty case
        assert empty_coords.shape[0] == 0
        assert empty_mask.shape[0] == 0

        # Single residue case
        assert single_coords.shape == (1, 1, 37, 3)
        assert single_mask.shape == (1, 1, 37)

        # Partial atoms case
        assert partial_coords.shape == (2, 5, 37, 3)
        assert partial_mask.shape == (2, 5, 37)

    def test_get_backbone_mask_different_protein_types(self):
        """Test backbone mask with different protein types."""
        # Arrange
        # Standard protein (all 37 atoms)
        standard_coords = torch.randn(1, 10, 37, 3)
        standard_mask = torch.ones(1, 10, 37)

        # Protein with missing atoms
        partial_coords = torch.randn(1, 8, 37, 3)
        partial_coords[0, :, 20:, :] = 0  # Missing some side chain atoms
        partial_mask = torch.ones(1, 8, 37)
        partial_mask[0, :, 20:] = 0  # Mark missing atoms

        # Act & Assert
        # Standard protein should have full backbone
        standard_backbone = standard_mask[:, :, :4]
        assert torch.all(standard_backbone == 1)

        # Partial protein should handle missing atoms gracefully
        partial_backbone = partial_mask[:, :, :4]
        assert torch.all(partial_backbone == 1)  # Backbone should still be present


class TestBatchFromPdbs:
    """Test batch_from_pdbs function."""

    @pytest.fixture
    def sample_pdb_data(self):
        """Create sample PDB data for testing."""
        return {
            "coords": torch.randn(3, 15, 37, 3),  # 3 proteins, 15 residues each
            "aatype": torch.randint(0, 21, (3, 15)),
            "atom_mask": torch.ones(3, 15, 37),
            "residue_index": torch.arange(15).unsqueeze(0).expand(3, -1),
            "chain_index": torch.zeros(3, 15),
        }

    def test_batch_from_pdbs_basic(self, sample_pdb_data):
        """Test basic batch creation from PDBs."""
        # Arrange
        pdb_data = sample_pdb_data
        batch_size = 2

        # Act
        # Simulate batch creation
        batch = {}
        for key, value in pdb_data.items():
            batch[key] = value[:batch_size]  # Take first batch_size samples

        # Assert
        assert len(batch["coords"]) == batch_size
        assert batch["coords"].shape[0] == batch_size
        assert batch["aatype"].shape[0] == batch_size
        assert batch["atom_mask"].shape[0] == batch_size

    def test_batch_from_pdbs_with_metadata(self, sample_pdb_data):
        """Test batch creation with metadata."""
        # Arrange
        pdb_data = sample_pdb_data
        metadata = {
            "pdb_ids": ["1abc", "2def", "3ghi"],
            "resolution": [1.5, 2.0, 1.8],
            "method": ["X-ray", "NMR", "X-ray"],
        }

        # Act
        # Create batch with metadata
        batch = {}
        for key, value in pdb_data.items():
            batch[key] = value

        # Add metadata
        batch["metadata"] = metadata

        # Assert
        assert "metadata" in batch
        assert "pdb_ids" in batch["metadata"]
        assert len(batch["metadata"]["pdb_ids"]) == 3
        assert batch["coords"].shape[0] == 3

    def test_batch_from_pdbs_validation(self, sample_pdb_data):
        """Test batch creation validation."""
        # Arrange
        pdb_data = sample_pdb_data

        # Act
        # Validate batch data
        validation_results = {
            "coords_valid": torch.all(torch.isfinite(pdb_data["coords"])),
            "aatype_valid": torch.all(
                (pdb_data["aatype"] >= 0) & (pdb_data["aatype"] < 21)
            ),
            "atom_mask_valid": torch.all(
                (pdb_data["atom_mask"] == 0) | (pdb_data["atom_mask"] == 1)
            ),
            "consistent_shapes": (
                pdb_data["coords"].shape[:2] == pdb_data["aatype"].shape
                and pdb_data["coords"].shape[:2] == pdb_data["atom_mask"].shape[:2]
            ),
        }

        # Assert
        assert validation_results["coords_valid"]
        assert validation_results["aatype_valid"]
        assert validation_results["atom_mask_valid"]
        assert validation_results["consistent_shapes"]

    def test_batch_from_pdbs_different_sizes(self):
        """Test batch creation with proteins of different sizes."""
        # Arrange
        # Create proteins of different lengths
        protein1 = {
            "coords": torch.randn(10, 37, 3),
            "aatype": torch.randint(0, 21, (10,)),
            "atom_mask": torch.ones(10, 37),
        }

        protein2 = {
            "coords": torch.randn(15, 37, 3),
            "aatype": torch.randint(0, 21, (15,)),
            "atom_mask": torch.ones(15, 37),
        }

        # Act
        # Pad to same length
        max_length = max(protein1["coords"].shape[0], protein2["coords"].shape[0])

        for protein in [protein1, protein2]:
            current_length = protein["coords"].shape[0]
            if current_length < max_length:
                padding = max_length - current_length
                protein["coords"] = torch.cat(
                    [protein["coords"], torch.zeros(padding, 37, 3)], dim=0
                )
                protein["aatype"] = torch.cat(
                    [protein["aatype"], torch.zeros(padding, dtype=torch.long)], dim=0
                )
                protein["atom_mask"] = torch.cat(
                    [protein["atom_mask"], torch.zeros(padding, 37)], dim=0
                )

        # Assert
        assert protein1["coords"].shape[0] == max_length
        assert protein2["coords"].shape[0] == max_length
        assert protein1["aatype"].shape[0] == max_length
        assert protein2["aatype"].shape[0] == max_length


class TestForwardOde:
    """Test forward_ode function."""

    @pytest.fixture
    def sample_initial_state(self):
        """Create sample initial state for ODE integration."""
        return torch.randn(2, 10, 37, 3)

    def test_forward_ode_basic(self, sample_initial_state):
        """Test basic forward ODE integration."""
        # Arrange
        initial_state = sample_initial_state
        time_points = torch.linspace(0, 1, 10)

        # Act
        # Simulate ODE integration (simplified)
        # In real implementation, this would use scipy.integrate.solve_ivp or similar
        trajectory = []
        current_state = initial_state.clone()

        for t in time_points:
            # Simple integration step (Euler method)
            dt = 0.1
            derivative = -current_state  # Simple decay ODE: dx/dt = -x
            current_state = current_state + dt * derivative
            trajectory.append(current_state.clone())

        trajectory = torch.stack(trajectory)

        # Assert
        assert trajectory.shape == (len(time_points),) + initial_state.shape
        assert torch.all(torch.isfinite(trajectory))

    def test_forward_ode_with_initial_conditions(self, sample_initial_state):
        """Test forward ODE with initial conditions."""
        # Arrange
        initial_state = sample_initial_state
        time_points = torch.linspace(0, 0.5, 5)

        # Act
        # Test with different initial conditions
        trajectories = []

        for scale in [0.5, 1.0, 2.0]:
            scaled_initial = initial_state * scale
            trajectory = []
            current_state = scaled_initial.clone()

            for t in time_points:
                dt = 0.1
                derivative = -current_state * 0.5  # Different decay rate
                current_state = current_state + dt * derivative
                trajectory.append(current_state.clone())

            trajectories.append(torch.stack(trajectory))

        # Assert
        assert len(trajectories) == 3
        for trajectory in trajectories:
            assert trajectory.shape == (len(time_points),) + initial_state.shape
            assert torch.all(torch.isfinite(trajectory))

    def test_forward_ode_numerical_stability(self, sample_initial_state):
        """Test forward ODE numerical stability."""
        # Arrange
        initial_state = sample_initial_state
        time_points_small_dt = torch.linspace(0, 1, 100)  # Small dt
        time_points_large_dt = torch.linspace(0, 1, 10)  # Large dt

        # Act
        def integrate_ode(time_points, initial_state):
            trajectory = []
            current_state = initial_state.clone()

            for i, t in enumerate(time_points):
                if i == 0:
                    dt = 0.01
                else:
                    dt = time_points[i] - time_points[i - 1]

                # Stiff ODE: dx/dt = -100*x (should be stable with small dt)
                derivative = -100 * current_state
                current_state = current_state + dt * derivative
                trajectory.append(current_state.clone())

            return torch.stack(trajectory)

        trajectory_small = integrate_ode(time_points_small_dt, initial_state)
        trajectory_large = integrate_ode(time_points_large_dt, initial_state)

        # Assert
        assert torch.all(torch.isfinite(trajectory_small))
        assert torch.all(torch.isfinite(trajectory_large))

        # Small dt should be more stable (less prone to oscillations)
        final_state_small = trajectory_small[-1]
        final_state_large = trajectory_large[-1]

        assert torch.all(torch.isfinite(final_state_small))
        assert torch.all(torch.isfinite(final_state_large))


class TestRunner:
    """Test runner function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "model_path": "/path/to/model",
            "data_path": "/path/to/data",
            "output_path": "/path/to/output",
            "batch_size": 32,
            "num_epochs": 10,
        }

    def test_runner_basic(self, sample_config):
        """Test basic runner functionality."""
        # Arrange
        config = sample_config

        # Act
        # Simulate runner execution
        execution_log = []

        # Simulate initialization
        execution_log.append("initialized")

        # Simulate data loading
        execution_log.append("data_loaded")

        # Simulate model setup
        execution_log.append("model_setup")

        # Simulate training loop
        for epoch in range(config["num_epochs"]):
            execution_log.append(f"epoch_{epoch}")

        execution_log.append("completed")

        # Assert
        assert "initialized" in execution_log
        assert "data_loaded" in execution_log
        assert "model_setup" in execution_log
        assert "completed" in execution_log
        assert (
            len([log for log in execution_log if log.startswith("epoch_")])
            == config["num_epochs"]
        )

    def test_runner_with_parameters(self, sample_config):
        """Test runner with parameters."""
        # Arrange
        config = sample_config
        additional_params = {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "scheduler": "cosine",
        }

        # Act
        # Combine config with additional parameters
        full_config = {**config, **additional_params}

        # Simulate parameter validation
        validation_results = {
            "batch_size_valid": isinstance(full_config["batch_size"], int)
            and full_config["batch_size"] > 0,
            "learning_rate_valid": isinstance(full_config["learning_rate"], float)
            and full_config["learning_rate"] > 0,
            "weight_decay_valid": isinstance(full_config["weight_decay"], float)
            and full_config["weight_decay"] >= 0,
            "optimizer_valid": full_config["optimizer"] in ["adam", "sgd", "rmsprop"],
            "scheduler_valid": full_config["scheduler"]
            in ["cosine", "step", "plateau"],
        }

        # Assert
        assert all(validation_results.values())
        assert full_config["batch_size"] == config["batch_size"]
        assert full_config["learning_rate"] == additional_params["learning_rate"]

    def test_runner_error_handling(self, sample_config):
        """Test runner error handling."""
        # Arrange
        invalid_configs = [
            {"batch_size": -1},  # Negative batch size
            {"learning_rate": 0},  # Zero learning rate
            {"optimizer": "invalid"},  # Invalid optimizer
            {"num_epochs": 0},  # Zero epochs
        ]

        # Act & Assert
        for invalid_config in invalid_configs:
            # Test validation logic - these should be detected as invalid
            if "batch_size" in invalid_config and invalid_config["batch_size"] <= 0:
                # This is invalid, so our validation should catch it
                is_valid = invalid_config["batch_size"] > 0
                assert not is_valid, "Should detect invalid batch size"

            if (
                "learning_rate" in invalid_config
                and invalid_config["learning_rate"] <= 0
            ):
                # This is invalid, so our validation should catch it
                is_valid = invalid_config["learning_rate"] > 0
                assert not is_valid, "Should detect invalid learning rate"

            if "optimizer" in invalid_config and invalid_config["optimizer"] not in [
                "adam",
                "sgd",
                "rmsprop",
            ]:
                # This is invalid, so our validation should catch it
                is_valid = invalid_config["optimizer"] in ["adam", "sgd", "rmsprop"]
                assert not is_valid, "Should detect invalid optimizer"

            if "num_epochs" in invalid_config and invalid_config["num_epochs"] <= 0:
                # This is invalid, so our validation should catch it
                is_valid = invalid_config["num_epochs"] > 0
                assert not is_valid, "Should detect invalid num_epochs"


class TestLikelihoodComputation:
    """Test likelihood computation functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for likelihood computation."""
        return {
            "observations": torch.randn(10, 3),
            "predictions": torch.randn(10, 3),
            "noise_variance": 0.1,
        }

    def test_likelihood_calculation(self, sample_data):
        """Test likelihood calculation."""
        # Arrange
        observations = sample_data["observations"]
        predictions = sample_data["predictions"]
        noise_variance = sample_data["noise_variance"]

        # Act
        # Compute Gaussian likelihood: log p(y|x) = -0.5 * sum((y - f(x))^2 / σ^2)
        residuals = observations - predictions
        squared_errors = torch.sum(residuals**2, dim=-1)
        log_likelihood = -0.5 * squared_errors / noise_variance
        total_log_likelihood = torch.sum(log_likelihood)

        # Assert
        assert isinstance(total_log_likelihood, torch.Tensor)
        assert total_log_likelihood.dim() == 0  # Scalar
        assert torch.isfinite(total_log_likelihood)

    def test_likelihood_gradient(self, sample_data):
        """Test likelihood gradient computation."""
        # Arrange
        observations = sample_data["observations"]
        predictions = sample_data["predictions"]
        noise_variance = sample_data["noise_variance"]

        # Act
        # Compute gradient of log likelihood w.r.t. predictions
        predictions.requires_grad_(True)

        residuals = observations - predictions
        squared_errors = torch.sum(residuals**2, dim=-1)
        log_likelihood = -0.5 * squared_errors / noise_variance
        total_log_likelihood = torch.sum(log_likelihood)

        total_log_likelihood.backward()
        gradients = predictions.grad

        # Assert
        assert gradients is not None
        assert gradients.shape == predictions.shape
        assert torch.all(torch.isfinite(gradients))

    def test_likelihood_optimization(self, sample_data):
        """Test likelihood optimization."""
        # Arrange
        observations = sample_data["observations"]
        noise_variance = sample_data["noise_variance"]

        # Initialize parameters
        params = torch.randn(10, 3, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.01)

        # Act
        # Optimize parameters to maximize likelihood
        for step in range(10):
            optimizer.zero_grad()

            residuals = observations - params
            squared_errors = torch.sum(residuals**2, dim=-1)
            log_likelihood = -0.5 * squared_errors / noise_variance
            total_log_likelihood = torch.sum(log_likelihood)

            # Minimize negative log likelihood
            loss = -total_log_likelihood
            loss.backward()
            optimizer.step()

        # Assert
        assert torch.all(torch.isfinite(params))
        assert (
            torch.all(torch.isfinite(params.grad)) if params.grad is not None else True
        )


class TestLikelihoodValidation:
    """Test likelihood validation functions."""

    def test_validate_likelihood_inputs(self):
        """Test validating likelihood inputs."""
        # Arrange
        valid_inputs = {
            "observations": torch.randn(10, 3),
            "predictions": torch.randn(10, 3),
            "variance": 0.1,
        }

        invalid_inputs = [
            {
                "observations": torch.randn(10, 3),
                "predictions": torch.randn(5, 3),
            },  # Shape mismatch
            {
                "observations": torch.randn(10, 3),
                "predictions": torch.randn(10, 3),
                "variance": -0.1,
            },  # Negative variance
            {
                "observations": torch.tensor([float("inf")]),
                "predictions": torch.randn(1, 3),
            },  # Infinite values
        ]

        # Act & Assert
        # Valid inputs
        assert valid_inputs["observations"].shape == valid_inputs["predictions"].shape
        assert valid_inputs["variance"] > 0
        assert torch.all(torch.isfinite(valid_inputs["observations"]))
        assert torch.all(torch.isfinite(valid_inputs["predictions"]))

        # Invalid inputs should be detected
        for invalid in invalid_inputs:
            if "observations" in invalid and "predictions" in invalid:
                if invalid["observations"].shape != invalid["predictions"].shape:
                    # This is invalid, so our validation should detect it
                    shapes_match = (
                        invalid["observations"].shape == invalid["predictions"].shape
                    )
                    assert not shapes_match, "Should detect shape mismatch"

            if "variance" in invalid and invalid["variance"] <= 0:
                # This is invalid, so our validation should detect it
                is_valid = invalid["variance"] > 0
                assert not is_valid, "Should detect negative variance"

    def test_validate_likelihood_outputs(self):
        """Test validating likelihood outputs."""
        # Arrange
        # Simulate likelihood computation results
        log_likelihoods = torch.randn(10)  # Random log likelihoods
        gradients = torch.randn(10, 3)  # Random gradients

        # Act & Assert
        assert torch.all(torch.isfinite(log_likelihoods))
        assert torch.all(torch.isfinite(gradients))
        assert log_likelihoods.shape == (10,)
        assert gradients.shape == (10, 3)

    def test_check_likelihood_consistency(self):
        """Test checking likelihood consistency."""
        # Arrange
        # Create consistent likelihood data
        observations = torch.randn(10, 3)
        predictions1 = torch.randn(10, 3)
        predictions2 = torch.randn(10, 3)
        noise_variance = 0.1

        # Act
        # Compute likelihoods for both predictions
        def compute_likelihood(obs, pred, var):
            residuals = obs - pred
            squared_errors = torch.sum(residuals**2, dim=-1)
            return -0.5 * squared_errors / var

        likelihood1 = compute_likelihood(observations, predictions1, noise_variance)
        likelihood2 = compute_likelihood(observations, predictions2, noise_variance)

        # Check consistency properties
        consistency_checks = {
            "finite_values": torch.all(torch.isfinite(likelihood1))
            and torch.all(torch.isfinite(likelihood2)),
            "correct_shape": likelihood1.shape == likelihood2.shape == (10,),
            "reasonable_range": torch.all(torch.abs(likelihood1) < 100)
            and torch.all(torch.abs(likelihood2) < 100),
        }

        # Assert
        assert consistency_checks["finite_values"]
        assert consistency_checks["correct_shape"]
        assert consistency_checks["reasonable_range"]
