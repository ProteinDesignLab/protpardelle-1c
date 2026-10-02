# cc154 — making the extra-atom block learn bonded geometry

## The defect

cc152/cc153 at epoch 585 denoise canonical protein beautifully and the
extra-atom block badly. Measured over the 40-structure Sampleworks set, 8
samples each, σ=4, deviation of covalent bond lengths from the CCD ideal:

| bond category | n | meanDev | % >0.3 Å | % >0.5 Å |
|---|---|---|---|---|
| canonical C/N/O | 403,080 | 0.047 | 0.38% | 0.11% |
| canonical S (Met SD, Cys SG) | 2,811 | 0.073 | 1.64% | 0.53% |
| **extra-slot (37–60)** | **831** | **0.865** | **70.2%** | **54.8%** |

MSE (selenomethionine) is the cleanest illustration: 7 of its 8 heavy atoms sit
in canonical atom37 slots and are placed well; the single extra-block atom (SE,
slot 37) carries both of the side-chain-tip bonds and wrecks them —
crystal CG–SE 1.96 Å / SE–CE 1.97 Å versus samples at 2.33/1.53, 2.41/1.03,
1.10/1.46. 1.0 Å is not a chemical bond.

This is not a masking bug (SE has `mask=1, noncanon=1, is_ligand=0`, so it is
supervised) and not a parsing bug (the atom set round-trips exactly). It is also
the same defect as cc153's ligand failure — ligand atoms live *entirely* in the
extra block, which is why 71% of cc153's ligand bonds break. One root cause,
two symptoms.

Note the earlier reported figure of "0.2–0.3% of bonds broken" was C/N/O only:
the metric used a `d_ideal < 1.8 Å` cutoff to identify bonds, which silently
excluded C–S (1.81 Å) and C–Se (1.96 Å). The cutoff is now 2.2 Å.

## Why the extra block fails where atom37 succeeds

For a **canonical** slot, the slot index *is* the atom identity, so the bond
graph is constant across the entire dataset: slot 5 (CG) is 1.52 Å from slot 3
(CB) in every training example that has both. The model never has to represent
connectivity — it is baked into the slot semantics, and plain per-atom MSE is
enough to learn it.

For an **extra** slot, neither the element nor the bond partners are fixed.
Slot 37 is a selenium bonded to slots 5 and 19 in MSE, a carbon bonded to slot
35 in KCX, an oxygen bonded to slot 11 in PCA. The model must *infer* the bond
graph per component. The only feature that could tell it is the reference
conformer — and the ablation showed the reference conformer is the one channel
the model does not use (+3.6% when zeroed, versus +28% for the element one-hot).

That is the key insight, and it is an architecture problem, not a data problem:

> The reference conformer supplies ideal **coordinates**. Turning those into a
> bond length requires computing `|ref_a − ref_b|` — a pairwise, quadratic
> operation. The patch embedding is a single `Linear(61×22 → 256)` over the
> flattened slots, and the transformer then attends **between tokens**. Nothing
> in the architecture computes distances between slots *within* a token. The
> information is present in a form the network cannot cheaply use.

So feeding coordinates and hoping the net derives distances from them is the
wrong ask. Feed the distances.

## cc154: the change

Two additions, both in the cc152 idiom (per-atom channels concatenated into the
patch embedding; trunk, task and noise schedule untouched).

### 1. Bond-topology input channels (3 per atom)

Precomputed, rotation-invariant scalars — exactly the quantities the model needs
to reproduce, rather than coordinates it would have to differentiate:

| channel | meaning |
|---|---|
| `d_parent` | ideal distance to this atom's parent in a BFS spanning tree of the component's bond graph, rooted at the token anchor (CA for a residue, the anchor atom for a ligand). Scaled by 1/2 Å. |
| `d_anchor` | ideal distance to the token anchor. Scaled by 1/10 Å. |
| `n_bonds` | number of covalent bonds, scaled by 1/4. |

The spanning tree is enough to *build* the component; `d_anchor` pins it
relative to the token's position; `n_bonds` distinguishes a terminal atom from a
branch point. Computed for every component including the standard 20, so the
channels remain a deterministic function of `aatype` for canonical protein and
the zero-init equivalence argument still holds.

Ring-closure bonds are deliberately *not* in the tree — they are covered by the
loss below, which uses the full bond list.

### 2. Bonded-distance loss

The Karras-weighted per-atom MSE gives no gradient that distinguishes "both
atoms 0.5 Å off in the same direction" (bond intact) from "0.5 Å off in
opposite directions" (bond destroyed). An explicit term does:

```
L_bond = mean over bonded pairs (i,j) of ( |x_i - x_j| - d_ideal(i,j) )^2
```

applied to the predicted `x_0`, over the CCD bond list, restricted to pairs
where at least one partner is in the structure loss (so cc152's clamped ligands
contribute nothing). Weighted by `bond_loss_weight`, warmed up over the first
`bond_loss_warmup` steps so the fine-tune is not destabilised by a large new
gradient landing on the extra-slot output rows.

### Deliberately deferred

* **Cross-token bond pair-bias in attention.** The trunk already accepts
  `pair_bias` (used by `ssadj`), so a bond-adjacency bias between tokens is
  cheap. But it only buys the bonds *between* tokens — components split over
  more than 24 atoms, and glycan chains. Both MSE bonds and most PTM bonds are
  intra-token, so this is not where the 70% failure lives. Follow-up, not cc154.
* **Internal-coordinate prediction** (predict bond/angle/torsion for extra
  atoms, guaranteeing correct lengths by construction). Strongest fix, but it
  replaces the Cartesian diffusion parameterisation for those atoms. Too
  invasive to combine with a diagnosis this fresh.

## Fine-tune cc152, not from scratch

Recommendation: initialise cc154 from **cc152 epoch 586**, not from cc89.

* The canonical pathway is already excellent (0.047 Å, 0.38% broken) and cost
  ~160M training examples to get there. Nothing about the diagnosis suggests it
  is wrong; re-deriving it would waste a week.
* The three new channels zero-initialise exactly as cc152's did, so
  **cc154 at init is numerically identical to cc152 epoch 586**, and the only
  new gradient is the bond loss plus the new channels — aimed precisely at the
  broken pathway.
* It makes the experiment fast to falsify: the extra-slot bond metric should
  move within ~20 epochs (~6 h). From scratch we would not know for a week
  whether the idea works.

The argument for from-scratch would be that 586 epochs have entrenched a bad
extra-slot solution. That is possible but unlikely to be a barrier: the failure
looks like *absent* signal rather than a confidently wrong one (sample-to-sample
CG–SE scatters over 1.10–2.41 Å, i.e. high variance, not a sharp wrong mode).

**Decision rule:** if `extra-slot %>0.3 Å` has not fallen below ~30% (from 70%)
by epoch 610, stop and retrain from the cc89 init instead.
