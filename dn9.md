# DN9 Hypothesis Allocation — Ln-Binding Scaffold Design

**Budget:** 190 designs, single round, no iteration.
**Goal:** maximize coverage of distinct *hypotheses* about how a Ln-binding site can be built, anchored on Architector idealized geometries observed in natural LanM/LanD/LazR173 EF-sites. Avoid the structure-prediction oracle trap by not using AF2/ESMfold as a selection objective.

Source of priors: `tables/tip_atoms_best_matches.tex` — per-EF-site closest Architector geometries for LazR173 (CN7, Lu), 8FNS (CN9, Nd), 8FNR (CN9 + CN7, Dy), 8DQ2 (CN10 La + CN7 Na).

---

## 1. Design philosophy

- **No structure-prediction oracle in the objective.** Structure prediction is permitted only as a *post-hoc weak sanity filter* (drop obvious unfoldables, e.g. mean pLDDT < 50). It does not steer the search, so no adversarial-example accumulation.
- **Architector geometries are the primary scaffolding prior**, sampled in proportion to their natural frequency in the four reference scaffolds, with a scattering arm for rare geometries to preserve diversity.
- **Diversity comes from orthogonal axes**, not from running the same design multiple times.
- **Filters are hard constraints, not soft scoring.** Anything failing the filter stack is dropped before the 190-shot budget is spent.

---

## 2. Filter stack (hard constraints)

Applied to every candidate before counting against the 190 budget:

| Filter | Threshold | Rationale |
|---|---|---|
| pI | < 5 | Acidic scaffold, matches natural Ln-binding loops |
| K + E content | < 30% (combined) | Cap on long-chain charged residues (surface charge complexity) |
| Normalized RoG | < 4 | Globular, compact |
| Triple amino acids (XXX) | not allowed | Avoid low-complexity / repeat regions |
| Dunbrack rotamer score | scored, not gated | Rank by lowest rotameric strain within each cell |
| Ramachandran / omega | gross outliers only | Cheap non-oracle backbone-quality check |
| Structure prediction (sanity) | mean pLDDT > 50 (post-hoc) | Drop obvious unfoldables only — NOT used to rank or select |

**MANDATORY PRE-FLIGHT:** Run this full filter stack on the four natural reference scaffolds (LazR173, 8FNS, 8FNR, 8DQ2) **before** running on any design. If any natural scaffold fails the filter, the filter is wrong and must be relaxed — not the design. Particularly likely to fail: pI<5 + K+E<30% may exclude E-rich LanM/LanD-family sequences. Adjust thresholds before committing budget.

---

## 3. Diversity axes (orthogonal, factorial)

Three primary axes after pruning correlated ones:

### Axis A — Architector geometry (5 levels + scatter)
Weighted by natural frequency in the table.

| Geometry | CN | Shots |
|---|---|---|
| `am_c3_7H2O_c0` | 7 | 35 |
| `pentagonal_bipyramidal` | 7 | 30 |
| `am_c3_9H2O_c0` | 9 | 30 |
| `tricapped_trig_prism` | 9 | 25 |
| `axial_bicapped_hex_planar` | 10 | 35 |
| **Scatter** (`capped_octahedral` CN7, `cn9_YICLED`, `cn10_GAKDAY`) | mixed | 35 total (~12 each) |
| **Total** | | **190** |

### Axis B — Coordination strategy (3 levels)
How the CN budget is split between sidechain Os, backbone carbonyls, and waters.

| Strategy | Description |
|---|---|
| **Pure sidechain** | All CN positions from D/E carboxylates |
| **EF-hand-like** | CN−2 sidechain Os + 1 backbone C=O + 1 water/Thr-OH |
| **Backbone-heavy** | CN−3 (or more) from backbone Os and waters; few D/E sidechains |

### Axis C — Solvent exposure of binding site (3 levels)
Buried / partially exposed / surface-exposed. Affects Ln vs Ca²⁺ selectivity.

### Dropped axis — SS-flanking
Originally proposed but correlated with solvent exposure (helix-flanked → more buried, loop-flanked → more exposed). Dropped to keep cell counts populated. Record SS-flanking type post-hoc as a descriptor, not a design axis.

---

## 4. Cell layout

5 main geometries × 3 coordination strategies × 3 exposure = **45 main cells**.
Plus 3 scatter geometries × (subset of strategies/exposures) = ~10 scatter cells.
Total ≈ **55 cells**, ~3-4 shots per main cell.

For the scatter arm (35 shots across 3 rare geometries), do not attempt full factorial — pick the 1-2 strategy/exposure combinations that look most plausible per geometry and put ~4 shots there.

---

## 5. Per-shot tiebreak ranking

Within a cell, when more candidates pass filters than the cell can hold:

1. Lowest Dunbrack rotamer strain summed over coordinating residues
2. Backbone quality (Ramachandran / omega outlier count)
3. Tip-atom RMSD to the idealized Architector geometry (lower is better)

Do NOT use predicted structure confidence (pLDDT, pTM) as a tiebreak — it would re-introduce the oracle.

---

## 6. Open questions deferred to runtime

- Exact pI calculator (EMBOSS vs ProtParam vs ipc2) — pick one and stick with it.
- Whether RoG normalization is by N^(1/3) or by some Flory exponent — pick one and document.
- Scaffold length range — not yet specified; constrain to match the four reference scaffolds (~70-200 residues) unless deliberately exploring.
- How to operationalize "coordination strategy" at the design tool level (constraint files, motif specs) — tool-dependent.

---

## 7. What this allocation deliberately does NOT do

- Does not use structure prediction in the objective or ranking.
- Does not include SS-flanking as an independent axis (collapsed into exposure).
- Does not allocate evenly across geometries — natural frequency matters more than nominal diversity.
- Does not iterate; the 190 shots are committed in one round.
