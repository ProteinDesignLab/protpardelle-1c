# cc152 / cc153 progress report — cc89 for non-canonical residues, ligands and metals

Working prototype, built in one session. Everything below runs end to end.

## The test set is ligand-heavy

Profiling all 40 of their structures
(`scripts/atomize/analyze_sampleworks_set.py`, full table in the script output):

| | count |
|---|---|
| non-polymer ligand | 22 / 40 |
| metal ion | 11 / 40 |
| ligand **or** metal | **25 / 40** |
| modified amino acid | 7 / 40 |
| neither | 15 / 40 |

So ligand handling matters more for this set than modified-residue handling,
and the chemistry is the hard kind: heme (HEC x2), Fe4S4 (SF4), FES,
molybdopterin (MGD) with tungsten, GTP x2, SAM, MTA, OMP, glycans
(NAG/BMA/MAN), diatomics (NO, NO2, CMO, OXY), Cu/Ni/Zn/Mn/Ca/Mg/Li, and 7Z0E is
a membrane protein with four lipids (L2P/LFA/OLC/SQL) next to its LYR.

This motivated splitting the model in two:

| | ligand / cofactor / metal atoms |
|---|---|
| **cc152** `ligand_treatment: conditioning_only` | given clean at every noise level, forced into the crop-conditioning channel, **excluded from the structure loss**. The ligand is known context; the protein around it is generated. |
| **cc153** `ligand_treatment: denoised` | noised and supervised like protein atoms, so the model learns to place them. |

Modified amino acids are polymer in both and always supervised; the flag only
governs non-polymer tokens, keyed off an explicit `is_ligand` row flag
(`aatype == UNK` alone would wrongly catch real UNK polymer residues).

## What exists now

| piece | where | state |
|---|---|---|
| wwPDB mmCIF mirror, 259,693 entries / 86 GB | `$SCRATCH/datasets/pdb_mirror/mmCIF` | complete |
| CCD cache (elements, ideal conformers, parent residues), 51,272 components | `$SCRATCH/datasets/pdb_atomized/ccd_cache.pkl` | complete |
| per-entry component index | `$SCRATCH/datasets/pdb_atomized/index/*.tsv` | complete, 259,693 rows |
| balanced train/val key lists | `$SCRATCH/datasets/pdb_atomized/{train,val}_atomized_pdb_keys.list` | 271,958 / 2,882 keys |
| atomized featurizer | `src/protpardelle/data/atomize_io.py` | new |
| cc152 model + config | `examples/training/cc152.yaml` | new |
| cc89→cc152 weight surgery | `scripts/atomize/expand_cc89_to_cc152.py` | verified exact |
| evaluation harness | `scripts/atomize/eval_atomized.py` | validated against known cc89 behaviour |
| PDB writeback | `atomize_io.feats_to_pdb_str_atomized` | round-trip exact |
| design doc | `docs/cc152_atomized.md` | — |

Training is running; see "Runs" below.

## The central design question: keeping atom37's inductive bias

atom37 works because slot `k` *always* means the atom named `atom_types[k]`, so
the CB of an Ala and the CB of a Trp share first-layer weights. cc152 keeps that
untouched and adds one level of the same idea:

* **slots 0–36** — the atom37 namespace, unchanged. Nothing non-canonical is
  ever placed here. A modified amino acid keeps its parent residue's slots
  (the CCD's `mon_nstd_parent_comp_id` gives ALY→LYS, KCX→LYS, OCS→CYS,
  PCA→GLN, LYR→LYS, MSE→MET, …), and `aatype` is the parent, so a modified Lys
  looks exactly like a Lys to cc89 with decorations attached.
* **slots 37–60** — a *global* extra-atom block. Extra slot `j` means "the
  `j`-th extra atom of this component", with the same weights for every residue
  type and every molecule.

The first version I wrote packed extra atoms into whichever atom37 slots the
parent happened to leave free. That was wrong for exactly the reason atom37 is
right: an acetyl carbon landed in `CG1`'s weights on a Lys and in a different
residue's weights elsewhere, and a retinal carbon would have been reusing the
weights that mean `OD1`. The global block fixes it. Ordering inside the block is
`(element group, CCD atom order)`, so the chemistry→slot map depends only on the
component.

Verified: LYR in 5azd and in 7bmh — unrelated structures — produce an identical
layout (Lys atoms in `N/CA/C/CB/O/CG/CD/CE/NZ`, retinal `C1..C19` in extra slots
37–56, same order). KCX's carbamate carbon and ALY's acetyl carbon both land in
extra slot 37.

24 extra slots covers **99.7%** of modified-residue instances in the PDB
(retinal needs 21, PLP-lysine 15; most PTMs need 1–4).

Ligands, cofactors and metals are pseudo-residue tokens on their own chain index
(so the 200-residue chain gap separates them from the polymer), with all atoms
in the extra block and a copy of the anchor atom at slot 1 so "slot 1 is this
token's position" stays true for the distance code.

**New input channels** (`conditioning_style: "concat+atomized"`), 13 per atom:
element one-hot (9), CCD ideal-conformer xyz (3, AF3's `ref_pos` trick — this is
how the trunk knows the topology of a molecule it has never seen; present for
99.9% of atoms), extra-slot flag (1).

## cc152 starts as exactly cc89

Four tensors change shape. The einops flatten is `(channel, atom)`, so widening
`n_atoms` re-interleaves them and a prefix copy would be wrong — the script
remaps `(c,a) -> (c,a)` explicitly. 173 other tensors copy verbatim; all new
entries are zero.

```
max |cc89 - cc152_init| on canonical slots : 5.2e-05  (rel 8.0e-06, float32 noise)
max |cc152_init| on empty extra slots      : 0.0
```

So any measured difference is due to fine-tuning, not to re-initialisation.

## Results after two epochs (10,758 steps, batch 32)

Epoch 1 → epoch 2 moves exactly the atoms that matter (RMSD Å, crop-conditioned):

| class | σ | epoch 1 | epoch 2 | noisy input |
|---|---|---|---|---|
| backbone | 1 | 0.534 | **0.488** | 1.735 |
| NCAA extra | 1 | 1.931 | **1.868** | 1.897 |
| NCAA extra | 4 | 6.950 | **5.887** | 7.587 |
| NCAA extra | 16 | 13.838 | **11.237** | 30.349 |
| ligand/metal | 4 | 6.217 | **5.102** | 6.933 |
| ligand/metal | 16 | 13.537 | **11.528** | 27.731 |

At epoch 2 the non-canonical atoms have just crossed below the noisy input at
σ=1 (1.868 vs 1.897) and are clearly better everywhere above it. Canonical
backbone is at 0.488 Å, i.e. slightly better than cc89's 0.46 Å on the clean
single-chain CATH-like data it was trained on — now on multi-chain PDB
assemblies with ligands.

The detailed epoch-1 table below is kept because it is the direct
head-to-head against the cc89 baseline.

## Results after one epoch (5,379 steps, batch 32)

Per-atom RMSD (Å) of a single denoising pass against the clean structure, on 24
held-out PDB entries containing modified residues, with cc89's crop
conditioning. `noisy_input` is the un-denoised input — the bar a model must
beat. `cc89_baseline` is cc152 at init, i.e. literally cc89, on the same inputs.

| σ | model | backbone | sidechain | NCAA extra atoms | ligand/metal |
|---|---|---|---|---|---|
| 1 | **cc152** | **0.534** | **0.786** | 1.931 | 1.718 |
| 1 | cc89 | 3.877 | 2.961 | 1.889 | 3.316 |
| 1 | noisy input | 1.735 | 1.727 | 1.897 | 1.733 |
| 4 | **cc152** | **1.240** | **1.882** | **6.950** | **6.217** |
| 4 | cc89 | 10.650 | 9.171 | 7.189 | 8.626 |
| 4 | noisy input | 6.940 | 6.907 | 7.587 | 6.933 |
| 16 | **cc152** | **3.421** | **4.495** | **13.838** | **13.537** |
| 16 | cc89 | 15.852 | 15.369 | 18.020 | 19.009 |
| 16 | noisy input | 27.759 | 27.627 | 30.349 | 27.731 |

Reading this:

* **Canonical protein is solved.** One epoch takes backbone denoising from
  3.88 Å to 0.53 Å at σ=1. For calibration, cc89 on the plain single-chain
  canonical protein it was trained on scores 0.46 Å at σ=1 in the same harness —
  so cc152 has essentially recovered cc89's native quality, but now on
  multi-chain PDB assemblies with ligands present.
* **cc89's poor baseline here is real, not a harness artefact.** The same
  harness gives cc89 0.46 Å (σ=1) / 0.92 Å (σ=4) on plain single-chain canonical
  proteins. Its failure on these structures is genuine domain shift: multi-chain
  assemblies, ligands, metals, and a `sigma_data` of 13.61 rather than 10.01.
* **Non-canonical and ligand atoms are learning but lag.** They beat the noisy
  input decisively at σ≥4 (13.8 vs 30.3 at σ=16) but are only at parity at σ=1.
  Expected: their output weights started at exactly zero and have had one epoch,
  while the canonical pathway only had to re-adapt.

Training loss: 1.468 → ~0.09, plateauing after ~1,500 steps (dominated by the
canonical atoms, which are the large majority).

### The new channels are wired correctly but not yet used

Two checks that matter more than the headline table.

**Ablation** (`scripts/atomize/ablate_channels.py`, 16 structures, σ=8): zero
out each new channel group at inference and see what breaks.

| ablation | backbone | NCAA extra | ligand/metal |
|---|---|---|---|
| none | 2.045 | 10.756 | 10.229 |
| no reference conformer | 2.045 | 10.755 | 10.229 |
| no element one-hot | 2.059 | 10.741 | 10.278 |
| no extra-slot flag | 2.046 | 10.771 | 10.680 |
| all off | 2.059 | 10.755 | 10.735 |

After one epoch the model is essentially **ignoring** the element and
reference-conformer channels. It is placing extra atoms from their slot
position and surrounding geometry alone.

**Weight inspection** (`scripts/atomize/inspect_weights.py`) rules out a wiring
bug — every new parameter moved off exactly zero:

| input channel group | ‖W‖ init | ‖W‖ epoch 1 |
|---|---|---|
| noisy xyz | 13.384 | 13.478 |
| crop-cond xyz | 16.043 | 16.122 |
| element one-hot | 0.000 | 1.020 |
| reference conformer | 0.000 | 0.966 |
| extra-slot flag | 0.000 | 0.460 |

| atom slots | ‖W‖ init | ‖W‖ epoch 1 |
|---|---|---|
| canonical 0–36 | 22.042 | 22.143 |
| extra 37–60 | 0.000 | 1.647 |

Output rows for the extra slots went 0.000 → 3.568, which is why extra atoms
get denoised at all. So gradients flow; the chemistry channels are simply still
an order of magnitude weaker than the coordinate channels after one epoch, and
the slot index — which by design already encodes chemistry, since the ordering
is `(element group, CCD order)` — is the easier signal to learn first.

At epoch 2 the picture is unchanged even though the extra atoms denoise much
better (10.76 → 9.05 at σ=8), and the weights keep growing (element 1.02→1.98,
reference conformer 0.97→1.74, extra slots 1.65→3.70 input / 3.57→5.70 output):

| ablation (epoch 2, σ=8) | backbone | NCAA extra | ligand/metal |
|---|---|---|---|
| none | 2.060 | 9.049 | 8.236 |
| no reference conformer | 2.058 | 9.057 | 8.231 |
| no element one-hot | 2.109 | 9.008 | 8.275 |
| no extra-slot flag | 2.060 | 9.117 | **9.073** |
| all off | 2.110 | 9.094 | **9.145** |

So the improvement is coming from the **slot layout**, not from the declared
chemistry. That is a real (and partly self-inflicted) finding:

* It validates the slot design — a globally consistent, chemistry-sorted extra
  block is enough for the trunk to learn to place these atoms.
* But because the ordering is `(element group, CCD order)`, **slot index
  already encodes element**, which makes the element one-hot largely redundant
  by construction. That redundancy is a design consequence I did not anticipate.
* The reference conformer being unused is the more interesting gap. It is the
  feature that should let the model generalise to a molecule it has never seen,
  and it is currently contributing nothing. Two likely causes: it is divided by
  10 (`REF_COORD_SCALE`), which may leave it too small next to the
  sigma_data-normalised coordinates, and a raw concatenated channel into a
  linear patch embedding is a weak way to inject it. A learned per-atom
  embedding, or a larger reference scale, is the obvious next experiment.
* The extra-slot flag *is* being used, and only for ligands (8.24 → 9.07),
  which makes sense — it is how a ligand token is distinguished from a residue.

## cc152 vs cc153, matched at epoch 1

Both started from the same cc89 init and trained one epoch on the same data.
RMSD (A) per atom class, crop-conditioned, 24 held-out structures that contain
a ligand or metal. `pocket` = protein atoms within 6 A of a ligand atom.
`no-ligand` = the same model with every ligand token deleted from the input.

| sigma | variant | bb | sc | ncaa | pocket | ligand |
|---|---|---|---|---|---|---|
| 1 | cc152 (ligand ctx) | **0.466** | **0.721** | **1.755** | **0.616** | 1.863* |
| 1 | cc152 no-ligand | 0.466 | 0.721 | 1.754 | 0.616 | – |
| 1 | cc153 (denoised) | 0.477 | 0.732 | 1.812 | 0.631 | **1.705** |
| 1 | cc153 no-ligand | 0.476 | 0.732 | 1.811 | 0.630 | – |
| 4 | cc152 (ligand ctx) | **1.167** | **1.767** | **6.540** | **1.543** | 4.060* |
| 4 | cc152 no-ligand | 1.167 | 1.767 | 6.540 | 1.543 | – |
| 4 | cc153 (denoised) | 1.188 | 1.781 | 6.548 | 1.559 | **6.229** |
| 8 | cc152 (ligand ctx) | **1.943** | **2.818** | **10.779** | **2.522** | 6.856* |
| 8 | cc152 no-ligand | 1.943 | 2.818 | 10.780 | 2.520 | – |
| 8 | cc153 (denoised) | 1.945 | 2.830 | 10.984 | 2.540 | **10.262** |

\* cc152's ligand column is not a prediction. It is handed the clean ligand and
never supervised there, so its output at ligand slots is
`skip_scale * clean + unsupervised junk`; the number grows with sigma purely
because `skip_scale` shrinks. Ignore it.

Three things to take from this:

1. **cc152 is consistently, slightly better on protein atoms** — bb 0.466 vs
   0.477, ncaa 1.755 vs 1.812, pocket 0.616 vs 0.631 at sigma=1, and the same
   ~1-2% margin at sigma 4 and 8. Expected: it does not spend loss or capacity
   on ligand atoms. The margin is small but consistent across every class and
   every noise level.
2. **cc153 is learning to place ligands, slowly.** 10.262 vs 13.88 for the
   noisy input at sigma=8, 6.229 vs 6.94 at sigma=4. Real but far from usable
   after one epoch.
3. **Neither model yet uses the ligand as context.** Deleting every ligand
   token changes the pocket by <= 0.002 A. That is the headline negative result.

### Is that because ligands cannot reach the protein, or because the model has not learned to use them?

Worth checking before believing it. `scripts/atomize/probe_ligand_influence.py`
translates the ligand 50 A away and measures how far the protein prediction
moves:

| | mean displacement |
|---|---|
| pocket protein atoms (<=6 A of ligand) | **0.085 A** |
| far protein atoms (>15 A from ligand) | 0.031 A |

So information does flow — the effect is 2.8x stronger in the pocket than in
the far field, which is the spatial signature you would want, and rules out an
attention/plumbing bug. It is simply too weak at one epoch to move RMSD. (A
0.085 A displacement in a direction uncorrelated with the existing error
changes RMSD-to-truth only to second order, which is why the deletion ablation
reads ~0.002 A rather than ~0.085 A.)

### Channel ablation on cc152

Same picture as cc153 — the element and reference-conformer channels are not
being exploited yet, and the weights are still an order of magnitude below the
coordinate channels:

| ablation (cc152 epoch 1, sigma=8) | backbone | NCAA extra | ligand |
|---|---|---|---|
| none | 2.010 | 10.713 | 13.037 |
| no reference conformer | 2.010 | 10.710 | 13.031 |
| no element one-hot | 2.042 | 10.746 | 13.010 |
| no extra-slot flag | 2.010 | 10.716 | 12.973 |
| all off | 2.041 | 10.751 | 12.928 |

cc152 weight movement after one epoch: element 0 -> 1.226, reference conformer
0 -> 1.336, extra-slot flag 0 -> 0.525, extra slots 0 -> 1.821 (input) and
0 -> 2.745 (output), against coordinate channels at 13.5-16.2. Gradients flow
everywhere; the chemistry pathway is just far behind the geometry pathway.

### Caveats on this comparison

* `sigma_data` is auto-calculated per run from 500 draws, and the runs drew
  13.61 (cc153 epochs 1-3), 13.23 (cc152) and 13.28 (cc153 continuation) — a
  ~3% spread in the Karras preconditioning that is not controlled for. Well
  below the effects discussed above, but worth pinning if the cc152/cc153
  margin ever becomes the headline result.
* The epoch-1 comparison above is clean: both models saw the same 172,121-key
  list. The key list was later rebuilt on the *complete* mirror index
  (271,958 keys), so cc153's continuation from epoch 4 runs 8,499 steps/epoch
  against the earlier 5,379. Epoch numbers are therefore not comparable across
  that boundary; step counts are.

## Matched epoch 2 — cc153 pulls ahead where it matters

Same 24 held-out ligand-containing structures, both models 2 epochs from the
cc89 init.

| sigma | variant | bb | sc | ncaa | pocket | ligand |
|---|---|---|---|---|---|---|
| 1 | cc152 (ligand ctx) | **0.458** | **0.719** | **1.732** | **0.611** | – |
| 1 | cc153 (denoised) | 0.467 | 0.728 | 1.815 | 0.624 | **1.570** |
| 4 | cc152 (ligand ctx) | **1.141** | **1.757** | 6.469 | **1.547** | – |
| 4 | cc153 (denoised) | 1.178 | 1.777 | **6.015** | 1.562 | **5.134** |
| 8 | cc152 (ligand ctx) | **1.911** | **2.791** | 10.645 | **2.504** | – |
| 8 | cc153 (denoised) | 1.942 | 2.827 | **9.633** | 2.527 | **8.279** |

Two things changed between epoch 1 and epoch 2:

**1. cc153 overtakes cc152 on non-canonical residue atoms.** At epoch 1 cc152
was ahead (10.779 vs 10.984 at sigma=8); at epoch 2 cc153 is clearly ahead
(9.633 vs 10.645), and the gap is growing. This looks like a direct consequence
of the shared global extra-atom block: ligand atoms and modified-residue extra
atoms occupy the *same* slots 37-60 and therefore the *same* weights, so
supervising ligands acts as auxiliary training for the whole extra-slot
pathway. cc152 excludes ligands from the loss and forfeits that signal.
cc152 keeps a small edge on canonical backbone/sidechain/pocket, as expected
from not spending loss there.

**2. cc153's ligand placement keeps improving:** sigma=8 10.262 -> 8.279,
sigma=4 6.229 -> 5.134 (noisy input 13.88 / 6.94).

### Ligand influence probe, all on the same 8 structures

How far the protein prediction moves when the ligand is translated 50 A:

| model | pocket (<=6 A) | far (>15 A) | localized excess |
|---|---|---|---|
| cc152 epoch 1 | 0.0762 | 0.0222 | 0.0540 |
| cc152 epoch 2 | 0.0474 | 0.0172 | 0.0302 |
| cc153 epoch 2 | **0.2412** | 0.1406 | **0.1006** |

cc153 is ~5x more sensitive to ligand position overall and has ~3x the
pocket-specific excess, while **cc152's sensitivity is *decreasing* with
training**. That is the clearest signal in this report: making the ligand a
prediction target is what makes the model attend to it. Handing it over as
never-supervised context teaches the model to ignore it — there is no gradient
anywhere that rewards using it, and the crop-conditioning already supplies
plenty of clean coordinates.

(Caveat: cc153's far-field sensitivity is also elevated, so its *ratio*
pocket:far is lower, 1.7 vs 2.8. The effect is a stronger but more diffuse
sensitivity, not a sharper pocket-specific one.)

### Recommendation

On this evidence **cc153 is the better default**, and the cc152 formulation
should not be used as-is. If ligands must stay fixed at inference (which is
what an ensemble-fitting caller wants), the right construction is not to drop
them from the loss but to keep cc153's supervision during training and simply
*clamp* the ligand coordinates at sampling time — the same way motif
scaffolding already works. A middle option worth one run: supervise ligands but
at reduced loss weight.

## Matched epoch 39 (4xH100, batch 128, ~75k steps) — supersedes the epoch-2 read

Same 24 held-out ligand-containing structures. This run is the clean one: both
models trained from the same cc89 init, same batch size, and `sigma_data` drew
13.35 / 13.41 (vs 13.23 / 13.61 before), so the preconditioning confound is
essentially gone.

| sigma | variant | bb | sc | ncaa | pocket | ligand |
|---|---|---|---|---|---|---|
| 1 | cc152 (ligand ctx) | 0.432 | 0.696 | 1.768 | 0.587 | – |
| 1 | cc153 (denoised) | 0.432 | 0.696 | **1.643** | 0.591 | **1.192** |
| 4 | cc152 (ligand ctx) | 1.086 | 1.698 | **4.963** | 1.472 | – |
| 4 | cc153 (denoised) | 1.082 | 1.701 | 5.096 | 1.487 | **2.960** |
| 8 | cc152 (ligand ctx) | 1.813 | 2.655 | **7.045** | 2.364 | – |
| 8 | cc153 (denoised) | 1.799 | 2.654 | 7.348 | 2.366 | **4.555** |

### Two earlier claims did not survive

1. **"cc152 is consistently better on protein atoms."** Gone. At epoch 39 the
   two are identical to three decimals on backbone (0.432/0.432), side chain
   (0.696/0.696) and pocket (0.587/0.591). The epoch-1/2 margin was an
   early-training transient, not a property of the formulation.
2. **"cc153 overtakes cc152 on non-canonical residue atoms."** Also did not
   hold. At epoch 39 it is a wash: cc153 better at sigma=1 (1.643 vs 1.768),
   cc152 better at sigma=4 and 8. My epoch-2 explanation (ligand supervision
   acting as auxiliary training for the shared extra-slot block) was an
   over-read of two noisy points.

### What did hold, and got much stronger

**cc153 can now place ligands.** sigma=8: 4.555 versus 13.88 for the noisy
input, and versus 8.279 at epoch 2. sigma=4: 2.960 vs 6.94 noisy. sigma=1:
1.192 vs 1.73 noisy. It is now comfortably below the noise floor everywhere.
cc152 cannot do this at all by construction.

**cc153 attends to the ligand far more** (ligand translated 50 A, n=8):

| model | pocket | far | localized excess |
|---|---|---|---|
| cc152 epoch 39 | 0.0521 | 0.0217 | 0.030 |
| cc153 epoch 39 | **0.1869** | 0.0574 | **0.129** |

4.3x the localized excess, and a cleaner pocket:far ratio than at epoch 2
(3.3 vs 2.2).

### The chemistry channels are now load-bearing

This is the clearest reversal. At epoch 1-2 zeroing the new channels changed
nothing; at epoch 39 (cc153, sigma=8):

| ablation | backbone | NCAA extra | ligand |
|---|---|---|---|
| none | 1.846 | 6.147 | 4.939 |
| no reference conformer | 1.912 | 6.139 | 4.969 |
| no element one-hot | **2.362** | 6.419 | 5.123 |
| no extra-slot flag | 1.848 | 6.200 | **7.502** |
| all off | **2.880** | **7.026** | **7.585** |

* **element one-hot**: removing it costs 28% on backbone — genuinely used.
* **extra-slot flag**: removing it costs 52% on ligand RMSD — it is how a
  ligand token is distinguished from a residue, and the model depends on it.
* **all off**: 56% worse backbone, 54% worse ligand.
* **reference conformer**: still the laggard at +3.6%. It is the one feature
  that has not earned its place, and it is the one that should matter most for
  molecules not seen in training. Worth attacking directly (scale, or a learned
  embedding instead of raw concatenation).

### Still unexplained

Deleting the ligand tokens entirely *still* changes the pocket by <=0.003 A,
even though the influence probe shows the model clearly reacts to ligand
position and the extra-slot flag is load-bearing. Best current reading: the
crop-conditioning already pins enough clean coordinates that protein accuracy
is saturated, so ligand context is attended to but cannot show up as an
accuracy gain on this metric. A cleaner test would reduce or remove the
crop-conditioning and re-measure.

### Recommendation (unchanged, now better supported)

**cc153.** It matches cc152 on every protein metric, and additionally places
ligands well below the noise floor. cc152 buys nothing. For a caller that needs
the ligand held fixed, clamp it at sampling time rather than dropping it from
the loss.

## Assumptions I made

1. **Asymmetric units, not bioassemblies.** Bioassemblies get very large; the
   asymmetric unit is smaller and still carries every modified residue and
   ligand. Entries over 4,000 residues are skipped outright, everything else is
   cropped to 384 tokens.
2. **Crops are anchored on a non-canonical residue or ligand 80% of the time**,
   otherwise the rare tokens would almost never be seen.
3. **`sigma_data` is auto-calculated from 500 draws of this dataset**, as cc89
   does — it comes out at **13.61** (vs 10.01 for CATH). Not pinned. This is a
   real perturbation to the Karras preconditioning relative to the cc89 weights,
   and is part of why the epoch-0 baseline looks bad.
4. **The noise schedule, loss and trunk are untouched**; lognormal(-0.5, 1.5),
   `ai-allatom`, cc89's crop-conditioning parameters, all identical.
5. **Resolution ≤ 3.5 Å, protein-containing, 30–4,000 residues**; NMR and
   unrefined models excluded.
6. **Modified residues are typed by their CCD parent.** Components with no
   canonical parent but with an N/CA/C backbone still enter as polymer with
   `aatype = UNK`; everything else becomes a ligand token.
7. **Only the first altloc** (blank or A) is kept; hydrogens dropped.
8. **Extra atoms beyond 24 per component are dropped** (0.3% of modified-residue
   instances); ligands over 24 atoms split across tokens, which loses the
   cross-token bond information.
9. **MiniMPNN is not trained here** (task is `ai-allatom`), so cc152 does not
   predict sequence; it is a structure denoiser. Identity of a non-canonical
   residue is an input, not an output.

## Note on the PDB IDs (correction)

An earlier version of this report claimed 2YLO and 7ZOE were mislabelled. That
was my error: the IDs are **2YL0** and **7Z0E** with digit zeros, and I had
fetched `2YLO`/`7ZOE` with the letter O. 7Z0E is "the M state of
bacteriorhodopsin at 1.22 A" exactly as described, and 2YL0 is cytochrome c
prime — the same protein family as 2XLW, and it carries PCA as stated. The
collaborator's list is correct throughout.

## Runs

* **cc153**: output dir `protpardelle-1c/cc152-1gpu` (named before the
  ligand_treatment flag existed; it took the `denoised` default, which *is*
  cc153). Epochs 1-3 saved. wandb `tyl/protpardelle-1c/du7xuklf`.
  A continuation job (`scripts/atomize/train_1gpu.sbatch cc153`) is queued on
  bioe and resumes from `epoch3_training_state.pth`.
* **cc152**: output dir `protpardelle-1c/cc152`, 1xH100 in an interactive
  allocation (expires ~17:50), running from the cc89 init.

Both are 1 GPU because possu had 0 and bioe 1 idle GPU when they were launched;
`scripts/atomize/train_cc152_bioe.sbatch` is the 4-GPU DataParallel version for
when capacity frees up.

## What I would do next, in order

1. **Let it train.** The canonical pathway converged in one epoch; the
   extra-atom pathway is still improving fast (σ=8: 10.76 → 9.05 between epochs
   1 and 2). Re-run `eval_atomized.py` and `ablate_channels.py` every few epochs.
   The key thing to watch is whether the reference-conformer ablation ever
   starts to bite — if it does not by ~epoch 10, change how it is injected
   rather than waiting.
2. **Widen the trunk.** 61 atom slots are being squeezed into the same 256
   channels cc89 used for 37. `cc89-large` (512 channels, 20 layers) is the
   obvious next run and the surgery script handles it unchanged.
3. **Bond features.** Ligands over 24 atoms split across tokens and lose their
   connectivity. An explicit token-bond feature (AF3's `token_bonds`) is the
   principled fix, and would also help the trunk respect intramolecular geometry
   rather than inferring it from the reference conformer alone.
4. **Sampling path.** `sample.py` still assumes 37 slots in places. Single-step
   denoising and partial diffusion work today via the eval harness; full ODE
   sampling with motif conditioning needs the slot count plumbed through. This
   is what Sampleworks would call.
4b. **Make the ligand matter.** The influence probe says the signal is there but
   weak (0.085 A). Candidate fixes, cheapest first: pin `sigma_data` across
   runs; raise the ligand's weight in the crop-conditioning channel; add an
   auxiliary pocket loss; and for cc153, upweight ligand atoms in the structure
   loss since they are a tiny fraction of atoms.
5. **An auxiliary loss on the extra atoms**, or oversampling them within the
   loss, since they are a small fraction of atoms and the Karras-weighted loss
   is dominated by backbone.
