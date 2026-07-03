import matplotlib.pyplot as plt
import numpy as np

from protpardelle.common.residue_constants import restype_atom37_mask, restypes, atom_types

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Original mask
im1 = ax1.imshow(restype_atom37_mask, cmap='viridis', aspect='auto', interpolation='nearest')
ax1.set_title('Original Atom Type Mask', fontsize=14, fontweight='bold')
ax1.set_xlabel('Atom Type', fontsize=12)
ax1.set_ylabel('Residue Type', fontsize=12)

# Set y-tick labels to residue names
ax1.set_yticks(range(20))
ax1.set_xticks(range(37))
ax1.set_xticklabels(atom_types)
ax1.set_yticklabels(restypes)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Mask Value', fontsize=12)

# Plot 2: Mask with Gaussian noise for zero entries
mask_with_noise = restype_atom37_mask.copy()

# Find zero entries and replace with Gaussian noise (mean=1, std=0.1)
zero_mask = (restype_atom37_mask == 0)
noise = np.random.normal(1.0, 0.1, size=zero_mask.sum())
mask_with_noise[zero_mask] = noise

im2 = ax2.imshow(mask_with_noise, cmap='viridis', aspect='auto', interpolation='nearest')
ax2.set_title('Mask with Gaussian Noise\n(Zero entries → N(1, 0.1²))', fontsize=14, fontweight='bold')
ax2.set_xlabel('Atom Type', fontsize=12)
ax2.set_ylabel('Residue Type', fontsize=12)

# Set y-tick labels to residue names
ax2.set_yticks(range(20))
ax2.set_xticks(range(37))
ax2.set_xticklabels(atom_types)
ax2.set_yticklabels(restypes)

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Mask Value', fontsize=12)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save as PDF
plt.savefig('atom_mask_visualization.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Plot saved as 'atom_mask_visualization.pdf'")

# Also save individual plots if needed
fig1, ax1_single = plt.subplots(figsize=(16, 8))
im1_single = ax1_single.imshow(restype_atom37_mask, cmap='viridis', aspect='auto', interpolation='nearest')
ax1_single.set_title('Original Atom Type Mask', fontsize=16, fontweight='bold')
ax1_single.set_xlabel('Atom Type', fontsize=14)
ax1_single.set_xticks(range(37))
ax1_single.set_xticklabels(atom_types)
ax1_single.set_ylabel('Residue Type', fontsize=14)
ax1_single.set_yticks(range(20))
ax1_single.set_yticklabels(restypes)
cbar1_single = plt.colorbar(im1_single, ax=ax1_single, shrink=0.8)
cbar1_single.set_label('Mask Value', fontsize=14)
plt.tight_layout()
plt.savefig('original_atom_mask.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Original mask saved as 'original_atom_mask.pdf'")

fig2, ax2_single = plt.subplots(figsize=(16, 8))
im2_single = ax2_single.imshow(mask_with_noise, cmap='viridis', aspect='auto', interpolation='nearest')
ax2_single.set_title('Atom Mask with Gaussian Noise\n(Zero entries → N(1, 0.1²))', fontsize=16, fontweight='bold')
ax2_single.set_xlabel('Atom Type', fontsize=14)
ax2_single.set_xticks(range(37))
ax2_single.set_xticklabels(atom_types)
ax2_single.set_ylabel('Residue Type', fontsize=14)
ax2_single.set_yticks(range(20))
ax2_single.set_yticklabels(restypes)
cbar2_single = plt.colorbar(im2_single, ax=ax2_single, shrink=0.8)
cbar2_single.set_label('Mask Value', fontsize=14)
plt.tight_layout()
plt.savefig('noise_filled_atom_mask.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("Noise-filled mask saved as 'noise_filled_atom_mask.pdf'")
