# cc152 / cc153 — cc89 for arbitrary biomolecular systems

Extends cc89 (atom37, `ai-allatom`) so that modified amino acids, ligands,
cofactors and metals are first-class inputs, while keeping the cc89 trunk,
task, loss and noise schedule untouched.

## Two variants

Identical in every respect except `model.ligand_treatment`:

| | ligand / cofactor / metal atoms |
|---|---|
| **cc152** `conditioning_only` | handed to the model **clean at every noise level** and **excluded from the structure loss**. The ligand is fixed, known context; the protein conformation around it is what gets generated. |
| **cc153** `denoised` | noised and supervised exactly like protein atoms, so the model learns to *generate* them. |

Modified amino acids are polymer in both cases and are always noised and
supervised — the flag only governs non-polymer pseudo-residue tokens
(`is_ligand`).

cc152 matches what an ensemble-fitting caller actually has: the
crystallographic ligand is given, the protein is the unknown. cc153 is the more
ambitious setting and is what you need if the ligand pose is also to be
designed.

Motivation: the Sampleworks group (Astera / Radial Science; Karson Chrispens,
Jaime Fraser) fits conformational ensembles to experimental data using
Protpardelle-1c/cc89 as the generative prior. Several of their test proteins
carry non-canonical residues — PCA (pyroglutamate), OCS (cysteinesulfonic
acid), KCX (carbamoyl-lysine), ALY (N6-acetyl-lysine), LYR (retinylidene-lysine
in bacteriorhodopsin) — which the atom37 loader silently drops.

Profiling their full 40-structure set (`scripts/atomize/analyze_sampleworks_set.py`)
shows ligands matter **more** than modified residues there:

| | count |
|---|---|
| non-polymer ligand | 22 / 40 |
| metal ion | 11 / 40 |
| ligand **or** metal | **25 / 40** |
| modified amino acid | 7 / 40 |
| neither | 15 / 40 |

and the chemistry is the hard kind: heme (HEC x2), Fe4S4 (SF4), FES,
molybdopterin (MGD) with tungsten, GTP x2, SAM, MTA, OMP, glycans (NAG/BMA/MAN),
diatomics (NO, NO2, CMO, OXY), Cu/Ni/Zn/Mn/Ca/Mg/Li, and 7Z0E is a membrane
protein with four lipids (L2P/LFA/OLC/SQL) alongside its LYR.

## Design

### Slot layout

cc89's strength is that atom37 slot `k` *always* means the atom named
`atom_types[k]`, so the CB of an Ala and the CB of a Trp share first-layer
weights. cc152 preserves that and adds one level of the same idea.

| slots | meaning |
|---|---|
| `0..36` | untouched atom37 namespace. Slot `k` == atom named `atom_types[k]`. Nothing non-canonical is ever placed here. |
| `37..60` | **global extra-atom block.** Extra slot `j` means "the `j`-th extra atom of this component", with the same weights for every residue type and every molecule. |

A modified amino acid keeps its parent residue's slots: the CCD's
`mon_nstd_parent_comp_id` gives ALY→LYS, KCX→LYS, OCS→CYS, PCA→GLN, LYR→LYS,
MSE→MET, …, and `aatype` is set to the parent. Every atom whose *name* is in
the atom37 vocabulary goes to its usual slot, so a modified Lys looks exactly
like a Lys to cc89 with extra atoms attached.

The extra atoms go into the shared block, **not** into whichever atom37 slots
the parent happens to leave free. That distinction matters: free-slot packing
would put an acetyl carbon into `CG1`'s weights when it decorates a Lys and
into some other residue's weights elsewhere, and would have a retinal carbon
reusing the weights that mean `OD1` — the opposite of the sharing atom37
exists to provide. Within the block, atoms are ordered by
`(element group, CCD atom order)`, so the chemistry→slot map is a property of
the component alone.

Verified consequence: LYR in 5azd and in 7bmh (unrelated structures) produce an
identical layout — Lys atoms in `N/CA/C/CB/O/CG/CD/CE/NZ`, retinal `C1..C19` in
extra slots 37–56 in the same order. KCX's carbamate carbon and ALY's acetyl
carbon both land in extra slot 37.

24 extra slots covers **99.7%** of modified-residue instances in the PDB
(retinal needs 21, PLP-lysine 15; most PTMs need 1–4).

Ligands, cofactors and metals become pseudo-residue tokens on their own chain
index, so the 200-residue chain gap in the relative positional encoding
separates them from the polymer. All their atoms live in the extra block; the
canonical slots stay empty apart from a copy of the anchor atom at slot 1,
which keeps "slot 1 is this token's position" true for the neighbour and
crop-conditioning distance code. Components with >24 atoms split over several
tokens.

### New input channels

`conditioning_style: "concat+atomized"` adds 13 per-atom channels
(9 + 13 = 22 channels × 61 slots = 1342 patch-embedding inputs):

* **element one-hot (9)** — `C N O S P Se halogen metal other`
* **reference conformer xyz (3)** — CCD *ideal* coordinates, centred per
  component and divided by 10. This is what lets the trunk know the internal
  topology of a molecule it has never seen (AF3's `ref_pos` trick). Present for
  99.9% of atoms.
* **extra-slot flag (1)**

For canonical protein these channels are a deterministic function of `aatype`,
so they add no information and the model can ignore them.

### Weight transfer

`scripts/atomize/expand_cc89_to_cc152.py` copies cc89 epoch415 into cc152.
Four tensors change shape because the einops flatten is `(channel, atom)` —
widening `n_atoms` re-interleaves them, so a prefix copy is wrong and the
script remaps `(c, a) -> (c, a)` explicitly:

```
to_patch_embedding.1.weight      (256, 333)  -> (256, 1342)   [(9,37) -> (22,61)]
cond_to_patch_embedding.1.weight (1024, 111) -> (1024, 183)
from_patch.1.weight              (111, 256)  -> (183, 256)
from_patch.1.bias                (111,)      -> (183,)
```

All 173 other tensors are copied verbatim. New entries are zero, so **cc152 at
init is numerically identical to cc89** — verified by
`scripts/atomize/test_cc89_equivalence.py`:

```
max |cc89 - cc152| on canonical slots : 5.2e-05  (rel 8.0e-06)   <- float32 noise
max |cc152| on empty extra slots      : 0.0
```

## Data

The full wwPDB mmCIF mirror (`rsync rsync.rcsb.org::ftp_data`), 259,693 entries
/ 86 GB at `/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF`. Asymmetric
units, not bioassemblies — smaller, and they still carry every modified residue
and ligand.

Filters: entry contains protein, resolution present and ≤ 3.5 Å, 30–4000
observed residues. 217,475 entries survive, split by component content:

| bucket | entries |
|---|---|
| contains a modified amino acid | 18,875 |
| contains a ligand/metal of interest | 126,244 |
| plain | 70,214 |

Modified-residue entries are oversampled ×4 → 271,958 training keys
(~28% NCAA-containing, ~46% ligand, ~26% plain). Validation is held out by PDB
id hash (2,882 keys) so it never overlaps training.

Crops: assemblies over 4000 residues are skipped outright; everything else is
cropped to `fixed_size` (384) tokens around an anchor that is a non-canonical
residue or ligand 80% of the time, so the rare tokens are actually seen.
Loading costs ~28 ms/example.

`sigma_data` is auto-calculated from 500 draws of this dataset as usual
(`auto_calc_sigma_data: True`) and comes out at **13.61**, vs 10.01 for
augmented-Ingraham CATH. It is *not* pinned to cc89's value.

## Pipeline

```bash
# 1. mirror the PDB (8-way rsync array, ~30 min)
sbatch scripts/atomize/rsync_pdb_mirror.sbatch

# 2. CCD cache: elements, ideal conformers, parent residues (~17 s)
python scripts/atomize/build_ccd_cache.py

# 3. index which components each entry contains (~4 min on 4x16 cores)
sbatch scripts/atomize/index_mirror.sbatch

# 4. filter + balance into train/val key lists
python scripts/atomize/build_train_index.py

# 5. cc89 -> cc152 weight surgery
python scripts/atomize/expand_cc89_to_cc152.py \
    --out /scratch/users/tianyulu/protpardelle-1c-experiments/cc152_init/cc152_from_cc89_epoch415.pth

# 6. checks
python scripts/atomize/test_cc89_equivalence.py
python scripts/atomize/test_featurizer.py

# 7. train
sbatch scripts/atomize/train_cc152.sbatch cc152

# 8. evaluate against the cc89 baseline
python scripts/atomize/eval_atomized.py --ckpt <ckpt> --n 40 --crop-cond
```

## Code changes outside the new files

Deliberately small:

* `core/models.py` — `CoordinateDenoiser` gains `has_atomized_cond` (+13
  channels) and three optional per-atom arguments, plumbed through
  `Protpardelle.forward`.
* `train.py` — passes `atom_elem` / `atom_ref` / `atom_noncanon` from the batch;
  pads the atom37-shaped backbone mask up to `n_atoms`.
* `data/dataset.py` — registers the three features, and one `get_item` branch
  that calls the atomized loader for `pdb_path` named `pdb_atomized`.

Nothing touches the noise schedule, the loss, the trunk, or any existing
conditioning mode.

## Getting structures back out

`atomize_io.feats_to_pdb_str_atomized(feats, coords)` writes a PDB with the real
CCD atom names for the extra block (`C15` of a retinal, `OQ1` of a carbamate),
skipping the ligand anchor duplicate. Round-trip on 5azd is exact (3032 atoms
in, 3032 out).
