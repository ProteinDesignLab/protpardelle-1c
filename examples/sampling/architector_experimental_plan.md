# Architector Geometry Sampling — Experimental Plan

Companion to `architector_geometries_indexed/` and `architector_geometries_unindexed/` sampling arrays (jobs `arch_idx` 22120941 and `arch_unidx` 22120944). Planning doc for turning ~97,000 raw designed structures into an orderable gene set for REE-binding experiments.

## Scope
- **379 motifs** across 16 REEs (Ce, Dy, Er, Eu, Gd, Ho, La, Li, Lu, Nd, Pm, Pr, Sm, Tb, Tm, Yb)
- **4 models**: cc91-tip-condmask (small), cc91-tip-condmask-large, cc138 (indexed); cc137 (unindexed)
- **64 samples per motif × 4 models ≈ 97,000 all-atom designs**

## RF3 precision — reality check

**Sub-0.1 Å geometry differences are not realistically predictable by RF3 (or any ML folder).**

- RF3 / AF3 backbone RMSD on good predictions is typically **0.5–1.5 Å**; all-atom sidechain RMSD is **1–2 Å**. Metal-coordinating residues are harder because training data is sparse.
- Crystal structures at 1.8–2.5 Å resolution place heavy atoms to **±0.2–0.5 Å**. MD of a folded protein at room temperature moves atoms **0.5–1.5 Å RMSF** on ns timescales. So 0.1 Å is *below both experimental resolution and physiological thermal motion*.
- Carboxylate O–O distances within a single ASP swing 0.1–0.2 Å with torsion/H-bond rearrangement; no prediction can resolve below that.
- **What RF3 *can* reliably discriminate**: correctly placed coordinating sidechains (≤ 1 Å from target), proper fold topology (TM > 0.8), high pLDDT (> 80), low PAE between motif and scaffold.
- The 0.07 Å tip-atom recovery numbers from our analysis **are an artifact of direct conditioning**, not a predictable design signal. Don't try to rank between designs that differ by < 0.3 Å — it's within prediction noise.

## Filtering pipeline

### Stage 1 — Self-consistency and physical validity (compute only)
1. ProteinMPNN: **8 sequences per design**
2. ESMFold each sequence. Compute per design:
   - `scRMSD` — designed sequence folded back, compared to generated structure
   - `pLDDT_mean` — ESMFold confidence
   - `motif_CA_RMSD` — predicted fold vs. generated motif, CA only
   - `motif_allatom_RMSD` — on the *conditioned* tip atoms
3. Thresholds: `scRMSD ≤ 2.0 Å`, `pLDDT_mean ≥ 80`, `motif_CA_RMSD ≤ 1.0 Å`, `motif_allatom_RMSD ≤ 1.5 Å`. Expect 60–80% attrition.

### Stage 2 — Motif geometry faithfulness
4. **Tip atom recovery** — run the coordinate-matching script we built (see earlier analysis). Mean < 0.3 Å, max < 0.6 Å.
5. **Coordination completeness** — every matched sidechain must still be ASP/GLU in the final designed sequence. For unindexed, `matched_motif_idx`/`matched_motif_aa3` in `scaffold_info.csv` already encodes this — enforce `all ASP/GLU`.
6. **Pocket accessibility** — SASA of each matched tip oxygen (OD1/OD2/OE1/OE2). Reject fully buried pockets (SASA = 0) since metal must access them.

### Stage 3 — Diversity (avoid near-duplicates)
7. `foldseek easy-cluster` on surviving designs with TM threshold 0.6, or CA pairwise RMSD cluster with cutoff 3 Å.
8. Keep 1–3 representatives per cluster, prioritizing highest pLDDT / lowest scRMSD within each.

### Stage 4 — REE selectivity prioritization
9. **Coordination number** drives size selectivity:
   - CN8 / CN9 → larger lanthanides (La, Ce, Nd — early 4f)
   - CN6 / CN7 → smaller lanthanides (Yb, Lu — late 4f)
   Order a balanced panel: 2–3 CN8 + 2–3 CN7 variants per REE.
10. **Second-sphere variation** — include both `acetate` (shorter ASP-like) and `propanoate` (longer GLU-like), and variants with explicit water ligands, to sample loop-length flexibility.
11. **Cross-REE specificity** — prioritize geometries unique to one REE over promiscuous ones. A geometry that exists for all 16 REEs is unlikely to confer selectivity; one constrained to e.g. La by size is the actual design target.

### Stage 5 — Final RF3 ranking (optional but recommended)
12. Run **RF3-AA with the target metal ion** (La³⁺, Yb³⁺, etc.) over Stage 4 survivors. Metrics:
    - Metal-to-scaffold `pAE < 10`
    - Matched residue `pLDDT > 85`
    - Metal-sidechain contact preserved in prediction
13. **Rank only on differences ≥ 2–3 score-unit points** (pLDDT or pAE). Within-noise differences aren't meaningful — fall back to other criteria (diversity, SASA, topology).

## Experimental funnel

### Tier 1 — Individual testing (~30 designs per REE, $1–3k)
- Top-of-stage-5 ranking.
- Clone, express, purify.
- Measure Kd by **ITC** against target REE + 2–3 off-target REE for selectivity.

### Tier 2 — Pooled display screen (200–500 designs, chip-synthesized)
- **Yeast or bacterial display**, 96-well luminescence assay using **Tb³⁺-DO3A** or **Eu³⁺ sensitization** at residue tyr/trp.
- **FACS** enrichment, **NGS** to identify winners.
- Cheap per design ($5–20 each), catches unexpected hits from designs that looked worse by compute filters.

### Tier 3 — Structural confirmation (5–10 hits)
- X-ray crystallography at 1.5–2.0 Å, or cryo-EM for larger constructs.
- Full lanthanide-series competition ITC for selectivity profile.

## What compute can meaningfully add beyond Stage 3

- **Selectivity prediction** is the high-value compute task, not "0.1 Å discrimination". Two routes:
  - **FEP / thermodynamic integration** with a lanthanide-compatible force field (e.g., Clavaguéra 12-6-4 LJ) — gives ΔΔG between REEs for binders that have converged structure.
  - **ML ligand-binding models** (e.g., structure-conditioned affinity predictors) — lower fidelity but run at scale across Stage 3 survivors.
- Selectivity output ranking maps directly to the downstream separation application (Nd vs. Dy permanent magnet recycling, heavy vs. light REE biomining, etc.).

## Practical numbers

| Stage | Count | Cost per |
|-------|------:|----------|
| Raw samples | 97,000 | compute |
| Stage 1 pass (~30% survive) | ~30,000 | compute |
| Stage 2 pass (~50% survive) | ~15,000 | compute |
| Stage 3 dedup (~1-in-20) | ~750 | compute |
| Stage 4 REE rebalance | ~300 | compute + judgement |
| Stage 5 RF3 rerank | ~300 | compute |
| Tier 1 order | ~30 × 16 REEs = 480 | $1–3k total |
| Tier 2 pool | 200–500 total | $2–5k |
| Tier 3 structures | 5–10 | per-structure |

---

*Generated 2026-04-20 alongside the architector sampling run launch.*
