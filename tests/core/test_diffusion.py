"""Tests for protpardelle.core.diffusion module."""

import pytest
import torch

from protpardelle.core.diffusion import compute_sampling_noise, noise_schedule


class TestComputeSamplingNoise:
    """Test the compute_sampling_noise function."""

    def test_basic_functionality(self):
        """Test basic functionality with default parameters."""
        timestep = torch.tensor(0.5)
        noise = compute_sampling_noise(timestep)

        assert isinstance(noise, torch.Tensor)
        assert noise.shape == timestep.shape
        assert noise.item() > 0

    def test_scalar_timestep(self):
        """Test with scalar timestep values."""
        timestep = torch.tensor(0.5)
        noise = compute_sampling_noise(timestep, sigma_data=10.0, s_min=0.001, s_max=80.0, rho=7.0)

        assert noise.shape == ()
        assert noise.item() > 0

    def test_1d_timestep(self):
        """Test with 1D tensor of timesteps."""
        timestep = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        noise = compute_sampling_noise(timestep)

        assert noise.shape == (5,)
        assert torch.all(noise > 0)

    def test_multidimensional_timestep(self):
        """Test with multi-dimensional timestep tensor."""
        timestep = torch.rand((2, 3, 4))
        noise = compute_sampling_noise(timestep)

        assert noise.shape == (2, 3, 4)
        assert torch.all(noise > 0)

    def test_timestep_zero(self):
        """Test with timestep = 0 (minimum noise)."""
        timestep = torch.tensor(0.0)
        noise = compute_sampling_noise(timestep, sigma_data=10.0, s_max=80.0)

        # At timestep 0, we should get minimum noise
        assert noise.item() > 0

    def test_timestep_one(self):
        """Test with timestep = 1 (maximum noise)."""
        timestep = torch.tensor(1.0)
        noise = compute_sampling_noise(timestep, sigma_data=10.0, s_min=0.001)

        # At timestep 1, we should get maximum noise
        assert noise.item() > 0

    def test_noise_increases_with_timestep(self):
        """Test that noise increases as timestep increases (timestep 1=high noise, 0=low noise)."""
        timesteps = torch.tensor([0.0, 0.5, 1.0])
        noises = compute_sampling_noise(timesteps)

        # Timestep convention: timestep=1 is high noise, timestep=0 is low noise
        # noise should increase as timestep increases
        assert noises[0] < noises[1] < noises[2]

    def test_different_sigma_data(self):
        """Test with different sigma_data values."""
        timestep = torch.tensor(0.5)
        noise1 = compute_sampling_noise(timestep, sigma_data=5.0)
        noise2 = compute_sampling_noise(timestep, sigma_data=10.0)
        noise3 = compute_sampling_noise(timestep, sigma_data=20.0)

        # Noise should scale with sigma_data
        assert noise1 < noise2 < noise3

    def test_different_rho(self):
        """Test with different rho values."""
        timestep = torch.tensor(0.5)
        noise1 = compute_sampling_noise(timestep, rho=1.0)
        noise2 = compute_sampling_noise(timestep, rho=7.0)

        # Different rho values should produce different noise levels
        assert noise1 != noise2

    def test_different_s_min_s_max(self):
        """Test with different s_min and s_max values."""
        timestep = torch.tensor(0.5)
        noise1 = compute_sampling_noise(timestep, s_min=0.001, s_max=80.0)
        noise2 = compute_sampling_noise(timestep, s_min=0.01, s_max=100.0)

        # Different ranges should produce different noise levels
        assert noise1 != noise2

    def test_batch_timesteps(self):
        """Test with batch of timesteps."""
        batch_size = 32
        timestep = torch.rand(batch_size)
        noise = compute_sampling_noise(timestep)

        assert noise.shape == (batch_size,)
        assert torch.all(noise > 0)


class TestNoiseSchedule:
    """Test the noise_schedule function."""

    def test_uniform_schedule(self):
        """Test uniform noise schedule."""
        timestep = torch.tensor([0.0, 0.5, 1.0])
        noise = noise_schedule(timestep, function="uniform")

        assert noise.shape == (3,)
        assert torch.all(noise > 0)
        # Should increase with timestep (timestep=1 is high noise, timestep=0 is low noise)
        assert noise[0] < noise[1] < noise[2]

    def test_lognormal_schedule(self):
        """Test lognormal noise schedule."""
        # Use timesteps that are valid for lognormal (between 0 and 1, exclusive)
        timestep = torch.tensor([0.1, 0.5, 0.9])
        noise = noise_schedule(
            timestep,
            function="lognormal",
            sigma_data=10.0,
            psigma_mean=-0.5,
            psigma_std=1.5,
        )

        assert noise.shape == (3,)
        assert torch.all(noise > 0)

    def test_mpnn_schedule(self):
        """Test MPNN noise schedule."""
        timestep = torch.tensor([0.0, 0.5, 1.0])
        noise = noise_schedule(timestep, function="mpnn", time_power=4.0)

        assert noise.shape == (3,)
        assert torch.all(noise >= 0)

    def test_constant_schedule(self):
        """Test constant noise schedule."""
        timestep = torch.tensor([0.0, 0.5, 1.0])
        constant_val = 5.0
        noise = noise_schedule(timestep, function="constant", constant_val=constant_val)

        assert noise.shape == (3,)
        assert torch.all(noise == constant_val)

    def test_constant_schedule_zero(self):
        """Test constant noise schedule with zero value."""
        timestep = torch.tensor([0.0, 0.5, 1.0])
        noise = noise_schedule(timestep, function="constant", constant_val=0.0)

        assert noise.shape == (3,)
        assert torch.all(noise == 0.0)

    def test_invalid_schedule_function(self):
        """Test that invalid schedule function raises ValueError."""
        timestep = torch.tensor(0.5)

        with pytest.raises(ValueError, match="Unknown noise schedule function"):
            noise_schedule(timestep, function="invalid")

    def test_uniform_with_custom_params(self):
        """Test uniform schedule with custom parameters."""
        timestep = torch.tensor(0.5)
        noise = noise_schedule(
            timestep,
            function="uniform",
            sigma_data=5.0,
            s_min=0.01,
            s_max=100.0,
            rho=5.0,
        )

        assert isinstance(noise, torch.Tensor)
        assert noise.item() > 0

    def test_lognormal_with_custom_params(self):
        """Test lognormal schedule with custom parameters."""
        timestep = torch.tensor(0.5)
        noise = noise_schedule(
            timestep,
            function="lognormal",
            sigma_data=15.0,
            psigma_mean=-1.0,
            psigma_std=2.0,
        )

        assert isinstance(noise, torch.Tensor)
        assert noise.item() > 0

    def test_mpnn_with_different_time_power(self):
        """Test MPNN schedule with different time_power values."""
        timestep = torch.tensor(0.5)
        noise1 = noise_schedule(timestep, function="mpnn", time_power=2.0)
        noise2 = noise_schedule(timestep, function="mpnn", time_power=4.0)
        noise3 = noise_schedule(timestep, function="mpnn", time_power=6.0)

        # Different time powers should produce different noise levels
        assert noise1 != noise2
        assert noise2 != noise3

    def test_multidimensional_timestep_uniform(self):
        """Test uniform schedule with multi-dimensional timesteps."""
        timestep = torch.rand((2, 3, 4))
        noise = noise_schedule(timestep, function="uniform")

        assert noise.shape == (2, 3, 4)
        assert torch.all(noise > 0)

    def test_multidimensional_timestep_lognormal(self):
        """Test lognormal schedule with multi-dimensional timesteps."""
        timestep = torch.rand((2, 3)) * 0.8 + 0.1  # Keep in (0.1, 0.9)
        noise = noise_schedule(timestep, function="lognormal")

        assert noise.shape == (2, 3)
        assert torch.all(noise > 0)

    def test_multidimensional_timestep_mpnn(self):
        """Test MPNN schedule with multi-dimensional timesteps."""
        timestep = torch.rand((4, 5))
        noise = noise_schedule(timestep, function="mpnn")

        assert noise.shape == (4, 5)
        assert torch.all(noise >= 0)

    def test_multidimensional_timestep_constant(self):
        """Test constant schedule with multi-dimensional timesteps."""
        timestep = torch.rand((3, 2))
        constant_val = 7.5
        noise = noise_schedule(timestep, function="constant", constant_val=constant_val)

        assert noise.shape == (3, 2)
        assert torch.all(noise == constant_val)

    def test_batch_processing(self):
        """Test with batch of timesteps for all schedule types."""
        batch_size = 16
        timestep = torch.rand(batch_size) * 0.8 + 0.1

        for function in ["uniform", "lognormal", "mpnn", "constant"]:
            noise = noise_schedule(timestep, function=function)
            assert noise.shape == (batch_size,)

    def test_edge_case_timestep_boundaries(self):
        """Test edge cases at timestep boundaries for uniform schedule."""
        timestep = torch.tensor([0.0, 1.0])
        noise = noise_schedule(timestep, function="uniform")

        assert noise.shape == (2,)
        assert torch.all(noise > 0)
        assert noise[0] < noise[1]  # Low noise at t=0, high noise at t=1

    def test_consistency_uniform_compute_sampling(self):
        """Test that uniform schedule matches compute_sampling_noise."""
        timestep = torch.tensor([0.0, 0.5, 1.0])
        sigma_data = 12.0
        s_min = 0.002
        s_max = 90.0
        rho = 6.0

        noise_from_schedule = noise_schedule(
            timestep,
            function="uniform",
            sigma_data=sigma_data,
            s_min=s_min,
            s_max=s_max,
            rho=rho,
        )

        noise_direct = compute_sampling_noise(
            timestep, sigma_data=sigma_data, s_min=s_min, s_max=s_max, rho=rho
        )

        assert torch.allclose(noise_from_schedule, noise_direct)
