from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.io import read, write, Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core import pretrained_mlip, FAIRChemCalculator


data_dir = Path("/home/users/tianyulu/code/protpardelle-1c/examples/motifs/omol")

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

# Configuration
pdb_files = [
    ("8FNS_EF2_EF3_LA_H2O.pdb", "La³⁺", "lightblue"),
    ("8FNS_EF2_EF3_ND_H2O.pdb", "Nd³⁺", "green"),
    ("8FNS_EF2_EF3_DY_H2O.pdb", "Dy³⁺", "orange"),
    ("8FNS_EF2_EF3_LU_H2O.pdb", "Lu³⁺", "purple"),
]
spin_states = [0, 6]  # Low-spin and high-spin configurations

# Initialize results storage
results = {
    "elements": [],
    "energies_spin0": [],
    "energies_spin6": [],
    "ionic_radii": [],
}

print("Computing energies for all REE complexes...\n")

# Loop over PDB files and spin states
for pdb_file, element, color in pdb_files:
    atoms = read(data_dir / pdb_file)
    
    energies = []
    for spin in spin_states:
        atoms.info.update({"spin": spin, "charge": 0})
        atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
        energy = atoms.get_potential_energy()
        energies.append(energy)
        print(f"{element} (spin={spin}): {energy:.4f} eV")
    
    results["elements"].append(element)
    results["energies_spin0"].append(energies[0])
    results["energies_spin6"].append(energies[1])
    
    # Ionic radii for 9-coordinate REE³⁺ (Angstroms)
    radii_map = {"La³⁺": 1.216, "Nd³⁺": 1.163, "Dy³⁺": 1.083, "Lu³⁺": 1.032}
    results["ionic_radii"].append(radii_map[element])
    
    print()

# Convert to arrays for easier manipulation
elements = np.array(results["elements"])
E_spin0 = np.array(results["energies_spin0"])
E_spin6 = np.array(results["energies_spin6"])
radii = np.array(results["ionic_radii"])

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("REE Binding Analysis for Protein EF-hand Loops", fontsize=16, fontweight='bold')

# 1. Absolute energies comparison
ax1 = axes[0, 0]
x = np.arange(len(elements))
width = 0.35
bars1 = ax1.bar(x - width/2, E_spin0, width, label='Spin=0 (Low)', alpha=0.8, color='steelblue')
bars2 = ax1.bar(x + width/2, E_spin6, width, label='Spin=6 (High)', alpha=0.8, color='coral')
ax1.set_xlabel('Rare Earth Element', fontsize=11, fontweight='bold')
ax1.set_ylabel('Potential Energy (eV)', fontsize=11, fontweight='bold')
ax1.set_title('Absolute Binding Energies', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(elements)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Relative binding energy (normalized to La³⁺)
ax2 = axes[0, 1]
E_rel_spin0 = E_spin0 - E_spin0[0]
E_rel_spin6 = E_spin6 - E_spin6[0]
ax2.plot(elements, E_rel_spin0, 'o-', linewidth=2.5, markersize=10, 
         label='Spin=0', color='steelblue')
ax2.plot(elements, E_rel_spin6, 's-', linewidth=2.5, markersize=10, 
         label='Spin=6', color='coral')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Rare Earth Element', fontsize=11, fontweight='bold')
ax2.set_ylabel('Relative Energy (eV vs La³⁺)', fontsize=11, fontweight='bold')
ax2.set_title('Relative Binding Preference', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Energy vs ionic radius (lanthanide contraction effect)
ax3 = axes[1, 0]
ax3.plot(radii, E_spin0, 'o-', linewidth=2.5, markersize=10, 
         label='Spin=0', color='steelblue')
ax3.plot(radii, E_spin6, 's-', linewidth=2.5, markersize=10, 
         label='Spin=6', color='coral')
for i, elem in enumerate(elements):
    ax3.annotate(elem, (radii[i], E_spin0[i]), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
ax3.set_xlabel('Ionic Radius (Å, 9-coordinate)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Potential Energy (eV)', fontsize=11, fontweight='bold')
ax3.set_title('Lanthanide Contraction Effect', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.invert_xaxis()  # Larger ions on left (La → Lu)

# 4. Selectivity metric: Energy differences
ax4 = axes[1, 1]
# Calculate pairwise selectivity (relative to Nd³⁺, a critical REE)
nd_idx = list(elements).index("Nd³⁺")
selectivity_spin0 = E_spin0 - E_spin0[nd_idx]
selectivity_spin6 = E_spin6 - E_spin6[nd_idx]

colors_map = {"La³⁺": "lightblue", "Nd³⁺": "green", "Dy³⁺": "orange", "Lu³⁺": "purple"}
colors = [colors_map[e] for e in elements]

bars = ax4.barh(elements, selectivity_spin0, alpha=0.7, color=colors)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Nd³⁺ reference')
ax4.set_xlabel('ΔE vs Nd³⁺ (eV, spin=0)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Rare Earth Element', fontsize=11, fontweight='bold')
ax4.set_title('Selectivity for Nd³⁺ (negative = preferred)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

# Add annotation
ax4.text(0.02, 0.98, 'More negative = Higher affinity\ncompared to Nd³⁺', 
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('ree_binding_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Binding Energy Analysis")
print("="*60)
print(f"{'Element':<10} {'Spin=0 (eV)':<15} {'Spin=6 (eV)':<15} {'ΔE vs Nd³⁺':<15}")
print("-"*60)
for i, elem in enumerate(elements):
    print(f"{elem:<10} {E_spin0[i]:<15.4f} {E_spin6[i]:<15.4f} {selectivity_spin0[i]:<15.4f}")
print("="*60)

# Additional selectivity metrics
print("\n" + "="*60)
print("SUGGESTED SELECTIVITY METRICS FOR EQUIMOLAR MIXTURE")
print("="*60)
print("""
Based on protein REE selectivity literature, consider computing:

1. **Binding Energy Difference (ΔE)**
   - Current metric: Relative energies vs reference ion
   - Lower energy = higher affinity and selectivity
   
2. **Coordination Geometry Metrics**
   - REE-O bond distances (mean, std deviation)
   - Coordination number (should be ~9 for REE³⁺)
   - Bite angle of carboxylate ligands
   - RMSD of oxygen coordination shell
   
3. **Structural Metrics**
   - Protein helicity change upon binding (CD spectroscopy analog)
   - Radius of gyration of binding pocket
   - Solvent accessible surface area of REE
   - Number of coordinating water molecules vs protein oxygens
   
4. **Thermodynamic Proxies**
   - Dehydration energy (fewer H2O = higher affinity)
   - Electrostatic complementarity score
   - Strain energy in protein backbone
   
5. **Selectivity Ratios**
   - α(Nd/Dy) = exp(-ΔΔE/kT) where ΔΔE = E_Nd - E_Dy
   - Pairwise selectivity for critical separations (Nd/Pr, Nd/Dy)

6. **Dynamic Properties** (if doing MD)
   - Residence time of coordinating waters
   - Flexibility of binding loop (B-factors)
   - Correlated motions between EF-hands
   
Key experimental correlations from literature:
- Higher charge density in binding loop → Higher affinity
- Ionic radius match to Ca²⁺ (~1.0 Å) → Optimal binding
- Complete dehydration → Stronger binding
- Aspartate-rich motifs → REE selectivity over Ca²⁺
""")

print("\nRecommended next steps:")
print("1. Extract coordination geometries from atoms object")
print("2. Calculate REE-O distances and coordination numbers")
print("3. Compute selectivity ratios: α = exp(-ΔE/kT) at 298K")
print("4. Compare with experimental Kd values if available")

