"""Pre-flight filter check on natural reference scaffolds for dn9.

Runs the dn9 filter stack on LazR173, 8FNS, 8FNR (no 8DQ2 available locally).
Reports per-scaffold pass/fail to decide whether thresholds need relaxation
before committing the 190-design budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dn9_filter import analyze_pdb, apply_filters

REFS = {
    "LazR173": "/home/users/tianyulu/code/rhaven/data/lazr173_xtal.pdb",
    "8FNS_A": "/home/users/tianyulu/code/rhaven/data/structure/8FNS_A.pdb",
    "8FNR_A": "/home/users/tianyulu/code/rhaven/data/structure/8FNR_A.pdb",
}

thresholds = {
    "pi_max": 5.0,
    "ke_max": 0.30,
    "rog_max": 4.0,
    "outlier_rate_max": 0.10,
}

results = {}
for name, path in REFS.items():
    p = Path(path)
    if not p.exists():
        print(f"[skip] {name}: {path} not found")
        continue
    d = analyze_pdb(p)
    if d is None:
        print(f"[skip] {name}: parse failed")
        continue
    passed, reasons = apply_filters(d, thresholds)
    d["passed"] = passed
    d["fail_reasons"] = reasons
    results[name] = d
    status = "PASS" if passed else "FAIL"
    print(f"\n=== {name} === [{status}]")
    print(f"  n_residues: {d['n_residues']}")
    print(f"  pI: {d['pi']:.2f} (must < {thresholds['pi_max']})")
    print(f"  K+E frac: {d['ke_frac']:.2%} (must < {thresholds['ke_max']:.0%})")
    print(f"  RoG/N^(1/3): {d['rog_norm']:.2f} (must < {thresholds['rog_max']})")
    print(f"  XXX triple: {d['has_xxx_triple']}")
    print(f"  Backbone outlier rate: {d['outlier_rate']:.2%} (must < {thresholds['outlier_rate_max']:.0%})")
    if not passed:
        print(f"  FAIL reasons: {reasons}")

# Summary
print("\n\n=== PREFLIGHT SUMMARY ===")
all_pass = all(r["passed"] for r in results.values())
if all_pass:
    print(f"All {len(results)} reference scaffolds passed. Thresholds are safe to use.")
else:
    fails = [n for n, r in results.items() if not r["passed"]]
    print(f"FAILED: {fails}")
    print("Suggest relaxing thresholds OR dropping the failing filter for problematic scaffolds.")

with open("/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/logs/preflight.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Wrote preflight results to dn9/logs/preflight.json")
