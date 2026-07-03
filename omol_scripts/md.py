"""
Molecular Dynamics Analysis of REE-Protein Binding
Computes dynamic properties: water residence time, binding site flexibility, 
and correlated motions between EF-hands
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase import units
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.io import read, write, Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ============================================================================
# Configuration
# ============================================================================
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

data_dir = Path("/home/users/tianyulu/code/protpardelle-1c/examples/motifs/omol")
pdb_files = [
    ("8FNS_EF2_EF3_LA_H2O.pdb", "La³⁺", "lightblue"),
    ("8FNS_EF2_EF3_ND_H2O.pdb", "Nd³⁺", "green"),
    ("8FNS_EF2_EF3_DY_H2O.pdb", "Dy³⁺", "orange"),
    ("8FNS_EF2_EF3_LU_H2O.pdb", "Lu³⁺", "purple"),
]

# MD parameters
temperature = 300  # K
timestep = 0.5  # fs
friction = 0.01 / units.fs  # 10 ps^-1, typical for protein-solvent systems
n_steps = 100000  # Total MD steps (2.5 ps total simulation time)
n_equilibration = 500  # Equilibration steps before analysis

# Analysis parameters
cutoff_distance = 3.5  # Angstroms, typical REE-O coordination distance
water_residence_cutoff = 2.8  # Angstroms, first hydration shell
analysis_interval = 10  # Analyze every N steps

# ============================================================================
# Helper Functions
# ============================================================================

def identify_ree_indices(atoms, ree_symbols=['La', 'Nd', 'Dy', 'Lu']):
    """Identify REE ion indices in the structure."""
    ree_indices = []
    for i, atom in enumerate(atoms):
        if atom.symbol in ree_symbols:
            ree_indices.append(i)
    return ree_indices

def identify_oxygen_indices(atoms):
    """Identify all oxygen atom indices."""
    return [i for i, atom in enumerate(atoms) if atom.symbol == 'O']

def identify_water_oxygens(atoms, protein_residues=None):
    """
    Identify water oxygen atoms (simplified: assume residue name or 
    atoms not in protein backbone C, N, Ca, etc.)
    """
    # Simplified: any O not bonded to C/N is likely water
    # In practice, you'd parse residue info from PDB
    water_o = []
    for i, atom in enumerate(atoms):
        if atom.symbol == 'O':
            # Simple heuristic: water O are isolated
            # Better: use atoms.info or parse PDB HETATM records
            water_o.append(i)
    return water_o

def get_coordination_shell(atoms, central_idx, oxygen_indices, cutoff=3.5):
    """Get indices of oxygens in coordination shell of central atom."""
    positions = atoms.get_positions()
    central_pos = positions[central_idx]
    oxygen_positions = positions[oxygen_indices]
    
    distances = np.linalg.norm(oxygen_positions - central_pos, axis=1)
    coordinating = np.array(oxygen_indices)[distances < cutoff]
    return coordinating.tolist(), distances[distances < cutoff]

def calculate_rmsf(trajectory, atom_indices):
    """
    Calculate Root Mean Square Fluctuation (RMSF) - analog of B-factors.
    RMSF measures flexibility of each atom around its average position.
    """
    positions = np.array([atoms.get_positions()[atom_indices] 
                          for atoms in trajectory])
    mean_positions = np.mean(positions, axis=0)
    deviations = positions - mean_positions
    rmsf = np.sqrt(np.mean(deviations**2, axis=0))
    return np.mean(rmsf, axis=1)  # Average over x,y,z

def calculate_water_residence_time(trajectory, ree_idx, water_indices, 
                                   cutoff=2.8, dt=0.5):
    """
    Calculate average residence time of water molecules in first shell.
    Uses continuous residence criterion.
    """
    residence_times = []
    
    for water_idx in water_indices:
        in_shell = []
        for atoms in trajectory:
            positions = atoms.get_positions()
            distance = np.linalg.norm(positions[water_idx] - positions[ree_idx])
            in_shell.append(distance < cutoff)
        
        # Find continuous residence events
        in_shell = np.array(in_shell)
        changes = np.diff(in_shell.astype(int))
        
        # Find entry and exit points
        entries = np.where(changes == 1)[0] + 1
        exits = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if in_shell[0]:
            entries = np.concatenate([[0], entries])
        if in_shell[-1]:
            exits = np.concatenate([exits, [len(in_shell)]])
        
        # Calculate residence times for each binding event
        for entry, exit in zip(entries, exits):
            residence_times.append((exit - entry) * dt)
    
    return np.array(residence_times)

def calculate_coordination_dynamics(trajectory, ree_idx, oxygen_indices, cutoff=3.5):
    """Track coordination number and geometry over time."""
    coord_numbers = []
    mean_distances = []
    
    for atoms in trajectory:
        coordinating, distances = get_coordination_shell(
            atoms, ree_idx, oxygen_indices, cutoff
        )
        coord_numbers.append(len(coordinating))
        if len(distances) > 0:
            mean_distances.append(np.mean(distances))
        else:
            mean_distances.append(np.nan)
    
    return np.array(coord_numbers), np.array(mean_distances)

def calculate_ef_hand_correlation(trajectory, ef1_indices, ef2_indices):
    """
    Calculate correlation between motions of two EF-hand loops.
    Uses center of mass motion correlation.
    """
    # Get center of mass trajectories for each EF-hand
    com1_traj = []
    com2_traj = []
    
    for atoms in trajectory:
        positions = atoms.get_positions()
        masses = atoms.get_masses()
        
        # EF-hand 1 COM
        com1 = np.average(positions[ef1_indices], 
                         weights=masses[ef1_indices], axis=0)
        com1_traj.append(com1)
        
        # EF-hand 2 COM
        com2 = np.average(positions[ef2_indices], 
                         weights=masses[ef2_indices], axis=0)
        com2_traj.append(com2)
    
    com1_traj = np.array(com1_traj)
    com2_traj = np.array(com2_traj)
    
    # Calculate displacement from average
    com1_fluct = com1_traj - np.mean(com1_traj, axis=0)
    com2_fluct = com2_traj - np.mean(com2_traj, axis=0)
    
    # Pearson correlation for each dimension
    correlations = []
    for dim in range(3):
        if np.std(com1_fluct[:, dim]) > 0 and np.std(com2_fluct[:, dim]) > 0:
            corr, _ = pearsonr(com1_fluct[:, dim], com2_fluct[:, dim])
            correlations.append(corr)
    
    return np.mean(correlations)

# ============================================================================
# Main MD Analysis Loop
# ============================================================================

results = {
    'elements': [],
    'avg_coord_number': [],
    'coord_number_std': [],
    'avg_ree_o_distance': [],
    'water_residence_time': [],
    'binding_site_rmsf': [],
    'ef_hand_correlation': [],
}

print("="*70)
print("MOLECULAR DYNAMICS ANALYSIS OF REE-PROTEIN COMPLEXES")
print("="*70)
print(f"Temperature: {temperature} K")
print(f"Timestep: {timestep} fs")
print(f"Total simulation: {n_steps * timestep / 1000:.2f} ps")
print(f"Equilibration: {n_equilibration * timestep / 1000:.2f} ps")
print("="*70 + "\n")

for pdb_file, element, color in pdb_files:
    print(f"\n{'='*70}")
    print(f"Analyzing {element}")
    print(f"{'='*70}")
    
    # Load structure
    atoms = read(data_dir / pdb_file)
    
    # Set spin state (high-spin for all REE³⁺)
    atoms.info.update({"spin": 6, "charge": 6})
    
    # Attach calculator
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    
    # Identify key atoms
    ree_indices = identify_ree_indices(atoms)
    oxygen_indices = identify_oxygen_indices(atoms)
    water_indices = identify_water_oxygens(atoms)
    
    print(f"Found {len(ree_indices)} REE ions")
    print(f"Found {len(oxygen_indices)} oxygen atoms")
    print(f"Found {len(water_indices)} water oxygens")
    
    # Set up Langevin dynamics
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=friction,
        fixcm=True  # Keep center of mass fixed
    )
    
    # Attach trajectory writer
    traj_save_dir = Path(f"/scratch/users/tianyulu/omol_outputs/uma_md/{n_steps}")
    traj_save_dir.mkdir(exist_ok=True, parents=True)
    traj_file = traj_save_dir / f"md_{element.replace('³⁺', '')}.traj"
    trajectory = Trajectory(traj_file, 'w', atoms)
    dyn.attach(trajectory.write, interval=analysis_interval)
    
    # Run equilibration
    print(f"\nRunning {n_equilibration} steps equilibration...")
    dyn.run(n_equilibration)
    
    # Run production MD
    print(f"Running {n_steps - n_equilibration} steps production MD...")
    trajectory_writer = trajectory
    dyn.run(n_steps - n_equilibration)
    trajectory.close()
    
    # Load trajectory for analysis
    print("\nAnalyzing trajectory...")
    traj = Trajectory(traj_file, 'r')
    
    # 1. Coordination number dynamics
    coord_nums, ree_o_dists = calculate_coordination_dynamics(
        traj, ree_indices[0], oxygen_indices, cutoff=cutoff_distance
    )
    avg_coord = np.mean(coord_nums)
    std_coord = np.std(coord_nums)
    avg_dist = np.nanmean(ree_o_dists)
    
    print(f"  Average coordination number: {avg_coord:.2f} ± {std_coord:.2f}")
    print(f"  Average REE-O distance: {avg_dist:.3f} Å")
    
    # 2. Water residence time
    if len(water_indices) > 0:
        residence_times = calculate_water_residence_time(
            traj, ree_indices[0], water_indices[:10],  # Analyze first 10 waters
            cutoff=water_residence_cutoff, dt=timestep * analysis_interval
        )
        avg_residence = np.mean(residence_times) if len(residence_times) > 0 else 0
        print(f"  Water residence time: {avg_residence:.3f} fs")
    else:
        avg_residence = 0
    
    # 3. Binding site flexibility (RMSF)
    # Get atoms within 5 Å of REE
    first_positions = traj[0].get_positions()
    ree_pos = first_positions[ree_indices[0]]
    nearby_atoms = np.where(
        np.linalg.norm(first_positions - ree_pos, axis=1) < 5.0
    )[0]
    
    rmsf_values = calculate_rmsf(traj, nearby_atoms)
    avg_rmsf = np.mean(rmsf_values)
    print(f"  Binding site RMSF: {avg_rmsf:.4f} Å")
    
    # 4. EF-hand correlation (if two EF-hands present)
    # Simplified: assume first half and second half of protein
    n_atoms_protein = len([a for a in atoms if a.symbol in ['C', 'N', 'O', 'S']])
    if n_atoms_protein > 20:
        ef1_indices = list(range(n_atoms_protein // 2))
        ef2_indices = list(range(n_atoms_protein // 2, n_atoms_protein))
        correlation = calculate_ef_hand_correlation(traj, ef1_indices, ef2_indices)
        print(f"  EF-hand motion correlation: {correlation:.3f}")
    else:
        correlation = 0
    
    # Store results
    results['elements'].append(element)
    results['avg_coord_number'].append(avg_coord)
    results['coord_number_std'].append(std_coord)
    results['avg_ree_o_distance'].append(avg_dist)
    results['water_residence_time'].append(avg_residence)
    results['binding_site_rmsf'].append(avg_rmsf)
    results['ef_hand_correlation'].append(correlation)

# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Molecular Dynamics Properties: REE Selectivity Indicators', 
             fontsize=16, fontweight='bold')

elements = results['elements']
colors_map = {"La³⁺": "lightblue", "Nd³⁺": "green", 
              "Dy³⁺": "orange", "Lu³⁺": "purple"}
colors = [colors_map[e] for e in elements]

# 1. Coordination number
ax = axes[0, 0]
bars = ax.bar(elements, results['avg_coord_number'], 
              yerr=results['coord_number_std'], alpha=0.7, color=colors)
ax.set_ylabel('Coordination Number', fontweight='bold')
ax.set_title('Average Coordination Number', fontweight='bold')
ax.axhline(y=9, color='red', linestyle='--', alpha=0.5, label='Ideal CN=9')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. REE-O distance
ax = axes[0, 1]
ax.plot(elements, results['avg_ree_o_distance'], 'o-', 
        markersize=12, linewidth=2.5, color='steelblue')
ax.set_ylabel('Distance (Å)', fontweight='bold')
ax.set_title('Average REE-O Bond Distance', fontweight='bold')
ax.grid(alpha=0.3)

# 3. Water residence time
ax = axes[0, 2]
bars = ax.bar(elements, results['water_residence_time'], alpha=0.7, color=colors)
ax.set_ylabel('Residence Time (fs)', fontweight='bold')
ax.set_title('Water Residence Time in 1st Shell', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.text(0.02, 0.98, 'Longer = Weaker binding', transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 4. Binding site flexibility (RMSF)
ax = axes[1, 0]
bars = ax.bar(elements, results['binding_site_rmsf'], alpha=0.7, color=colors)
ax.set_ylabel('RMSF (Å)', fontweight='bold')
ax.set_title('Binding Site Flexibility (B-factor analog)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.text(0.02, 0.98, 'Lower = More rigid = Higher selectivity', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 5. EF-hand correlation
ax = axes[1, 1]
bars = ax.bar(elements, results['ef_hand_correlation'], alpha=0.7, color=colors)
ax.set_ylabel('Correlation Coefficient', fontweight='bold')
ax.set_title('Correlated Motion Between EF-hands', fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim([-1, 1])
ax.grid(axis='y', alpha=0.3)
ax.text(0.02, 0.98, 'Higher = More cooperative binding', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 6. Combined selectivity score
ax = axes[1, 2]
# Normalize and combine metrics (lower RMSF + higher correlation = better)
norm_rmsf = 1 - (np.array(results['binding_site_rmsf']) / 
                 max(results['binding_site_rmsf']))
norm_corr = (np.array(results['ef_hand_correlation']) + 1) / 2  # Scale to 0-1
norm_residence = 1 - (np.array(results['water_residence_time']) / 
                     max(results['water_residence_time']))

selectivity_score = (norm_rmsf + norm_corr + norm_residence) / 3

bars = ax.barh(elements, selectivity_score, alpha=0.7, color=colors)
ax.set_xlabel('Combined Selectivity Score', fontweight='bold')
ax.set_title('Overall Dynamic Selectivity Indicator', fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(axis='x', alpha=0.3)
ax.text(0.02, 0.98, 'Higher = Better selectivity\n(rigid site + cooperative)', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('ree_md_dynamic_properties.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n" + "="*70)
print("SUMMARY: Dynamic Properties & Selectivity Metrics")
print("="*70)
print(f"{'Element':<10} {'Coord#':<10} {'REE-O(Å)':<12} {'H2O_res(fs)':<15} "
      f"{'RMSF(Å)':<12} {'EF-corr':<10}")
print("-"*70)
for i, elem in enumerate(elements):
    print(f"{elem:<10} {results['avg_coord_number'][i]:<10.2f} "
          f"{results['avg_ree_o_distance'][i]:<12.3f} "
          f"{results['water_residence_time'][i]:<15.1f} "
          f"{results['binding_site_rmsf'][i]:<12.4f} "
          f"{results['ef_hand_correlation'][i]:<10.3f}")
print("="*70)

print("""
INTERPRETATION GUIDE:
====================
1. **Coordination Number**: Should be ~9 for REE³⁺. Stable CN indicates good fit.

2. **REE-O Distance**: Shorter distances indicate stronger binding. Optimal ~2.3-2.5 Å.

3. **Water Residence Time**: LONGER time = WEAKER protein binding (water competes).
   - Shorter residence = water quickly replaced by protein ligands = higher affinity

4. **Binding Site RMSF (flexibility)**: LOWER = MORE RIGID = BETTER SELECTIVITY
   - Rigid binding sites are pre-organized for specific ion sizes
   - High RMSF indicates promiscuous binding

5. **EF-hand Correlation**: Higher positive correlation = cooperative binding
   - Indicates allosteric communication between binding sites
   - Important for multi-site selectivity

6. **Combined Score**: Integrates all dynamic metrics for overall selectivity ranking
""")
