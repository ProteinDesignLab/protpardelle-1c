"""Test flat-mode (no padding) vs padded-mode memory usage for per-atom model.

Run on an interactive GPU node:
    python scripts/test_flat_mode.py
"""

import gc
import sys
import time

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, "src")

from protpardelle.core.models import Protpardelle
from protpardelle.core import diffusion
from protpardelle.data.dataset import PerAtomPDBDataset, flat_collate_fn
from protpardelle.utils import load_config


def get_peak_memory_mb():
    """Return peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_memory():
    """Reset GPU memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_padded_forward_backward(model, batch, sigma_data):
    """Run padded-mode forward+backward, return peak memory."""
    reset_memory()
    device = next(model.parameters()).device

    coords = batch["atom_positions"].to(device)
    atom_mask = batch["atom_mask"].to(device)
    atom_element = batch["atom_element"].to(device)
    group_id = batch["group_id"].to(device)
    group_mask = batch["group_mask"].to(device)
    atom_charge = batch["atom_charge"].to(device)
    atom_mass = batch["atom_mass"].to(device)
    bs = coords.shape[0]
    n_groups = group_mask.shape[1]

    # Sample noise level
    timestep = torch.rand(bs, device=device).clamp(1e-9, 1 - 1e-9)
    noise_level = model.training_noise_schedule(timestep)

    # Noise coordinates
    noised_coords = diffusion.noise_coords_per_atom(coords, noise_level, atom_mask)

    # Forward
    denoised_coords, struct_self_cond, _, _ = model.forward_per_atom(
        noisy_coords=noised_coords,
        noise_level=noise_level,
        atom_mask=atom_mask,
        atom_element=atom_element,
        group_id=group_id,
        group_mask=group_mask,
        n_groups=n_groups,
        atom_charge=atom_charge,
        atom_mass=atom_mass,
        run_caliby_model=False,
    )

    # Loss
    struct_loss_mask = atom_mask.unsqueeze(-1).expand_as(coords)
    loss_weight = (noise_level ** 2 + sigma_data ** 2) / (
        (noise_level * sigma_data) ** 2
    )
    diff = (coords - denoised_coords) ** 2
    loss = (diff * struct_loss_mask).sum() / struct_loss_mask.sum().clamp(min=1)
    loss = loss * loss_weight.mean()

    # Backward
    loss.backward()

    peak = get_peak_memory_mb()
    return peak, loss.item()


def run_flat_forward_backward(model, flat_batch, sigma_data):
    """Run flat-mode forward+backward, return peak memory."""
    reset_memory()
    device = next(model.parameters()).device

    coords = flat_batch["atom_positions"].to(device)
    atom_element = flat_batch["atom_element"].to(device)
    group_id = flat_batch["group_id"].to(device)
    batch_idx = flat_batch["batch_idx"].to(device)
    atom_offsets = flat_batch["atom_offsets"].to(device)
    group_offsets = flat_batch["group_offsets"].to(device)
    atom_charge = flat_batch["atom_charge"].to(device)
    atom_mass = flat_batch["atom_mass"].to(device)
    bs = flat_batch["batch_size"]
    total_groups = group_offsets[-1].item()
    total_atoms = coords.shape[0]

    # Sample noise level (per sample)
    timestep = torch.rand(bs, device=device).clamp(1e-9, 1 - 1e-9)
    noise_level = model.training_noise_schedule(timestep)

    # Noise coordinates (flat: [total_atoms, 3])
    noise = torch.randn_like(coords) * noise_level[batch_idx].unsqueeze(-1)
    noised_coords = coords + noise

    # Forward (flat mode)
    denoised_coords, struct_self_cond = model.forward_per_atom_flat(
        noisy_coords=noised_coords,
        noise_level=noise_level,
        atom_element=atom_element,
        group_id=group_id,
        batch_idx=batch_idx,
        batch_size=bs,
        atom_offsets=atom_offsets,
        group_offsets=group_offsets,
        total_groups=total_groups,
        atom_charge=atom_charge,
        atom_mass=atom_mass,
    )

    # Loss (flat: all atoms are valid, no mask needed)
    diff = (coords - denoised_coords) ** 2
    loss = diff.sum() / total_atoms
    loss_weight = (noise_level ** 2 + sigma_data ** 2) / (
        (noise_level * sigma_data) ** 2
    )
    loss = loss * loss_weight.mean()

    # Backward
    loss.backward()

    peak = get_peak_memory_mb()
    return peak, loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load config
    config_path = "examples/training/cc91-peratom-cath.yaml"
    config = load_config(config_path)

    # Override checkpoint path (don't need to load weights)
    config.train.ckpt_path = ""
    config.model.pretrained_modules = []

    # Build model (random weights — we only care about memory)
    print("\nBuilding model...")
    model = Protpardelle(config, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    sigma_data = model.sigma_data

    # Load a small CATH dataset
    print("\nLoading dataset...")
    dataset = PerAtomPDBDataset(
        pdb_path="/scratch/users/tianyulu/augmented_ingraham_cath_bugfree",
        fixed_size=config.data.fixed_size,
        max_atoms_per_group=config.data.max_atoms_per_group,
        include_metals=False,
        mode="train",
        se3_data_augment=True,
        translation_scale=1.0,
        chain_residx_gap=200,
        subset="designable_",
    )
    print(f"Dataset size: {len(dataset)}")

    # Test batch sizes
    batch_sizes = [8, 16, 32, 48, 64]

    print("\n" + "=" * 70)
    print(f"{'BS':>4} | {'Mode':>8} | {'Total Atoms':>12} | {'Peak MB':>10} | {'Loss':>12} | {'Time':>8}")
    print("-" * 70)

    for bs in batch_sizes:
        # Load examples
        examples = []
        for i in range(bs):
            ex = dataset[i % len(dataset)]
            examples.append(ex)

        # Padded batch (default collate = stack)
        padded_batch = dataset.collate(examples)

        # Count actual atoms in padded batch
        total_padded_atoms = padded_batch["atom_mask"].shape[0] * padded_batch["atom_mask"].shape[1]
        real_atoms_padded = int(padded_batch["atom_mask"].sum().item())

        # Flat batch
        flat_batch = flat_collate_fn(examples)
        total_flat_atoms = flat_batch["atom_positions"].shape[0]

        # --- Test padded mode ---
        model.zero_grad()
        model.train()
        try:
            t0 = time.time()
            peak_padded, loss_padded = run_padded_forward_backward(model, padded_batch, sigma_data)
            dt_padded = time.time() - t0
            print(f"{bs:4d} | {'padded':>8} | {total_padded_atoms:>12d} | {peak_padded:>10.0f} | {loss_padded:>12.4f} | {dt_padded:>7.2f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:4d} | {'padded':>8} | {total_padded_atoms:>12d} | {'OOM':>10} | {'N/A':>12} | {'N/A':>8}")
            reset_memory()

        # --- Test flat mode ---
        model.zero_grad()
        model.train()
        try:
            t0 = time.time()
            peak_flat, loss_flat = run_flat_forward_backward(model, flat_batch, sigma_data)
            dt_flat = time.time() - t0
            print(f"{bs:4d} | {'flat':>8} | {total_flat_atoms:>12d} | {peak_flat:>10.0f} | {loss_flat:>12.4f} | {dt_flat:>7.2f}s")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:4d} | {'flat':>8} | {total_flat_atoms:>12d} | {'OOM':>10} | {'N/A':>12} | {'N/A':>8}")
            reset_memory()

        print("-" * 70)

        # Clean up
        del padded_batch, flat_batch, examples
        model.zero_grad()
        reset_memory()

    # Gradient sanity check on flat mode
    print("\n--- Gradient sanity check (flat mode, bs=8) ---")
    examples = [dataset[i % len(dataset)] for i in range(8)]
    flat_batch = flat_collate_fn(examples)
    model.zero_grad()
    model.train()
    reset_memory()

    _, loss = run_flat_forward_backward(model, flat_batch, sigma_data)
    n_grads = 0
    n_zero = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            n_grads += 1
            if p.grad.abs().max() == 0:
                n_zero += 1
        else:
            if p.requires_grad:
                print(f"  WARNING: {name} has no gradient!")
    print(f"  Parameters with gradients: {n_grads}")
    print(f"  Parameters with all-zero gradients: {n_zero}")
    print(f"  Loss: {loss:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
