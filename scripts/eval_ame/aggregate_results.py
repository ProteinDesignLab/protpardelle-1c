"""Aggregate per-sample metrics CSV into per-(model, motif) success rates.

Reads the output of compute_metrics.py and produces:
  - summary by model: total, success, pass rate
  - summary by (model, motif)
  - summary by motif (combined across models)
  - global success rate
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.metrics) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Per (model, motif)
    per_mm = defaultdict(lambda: {"n": 0, "succ": 0,
                                   "scrmsds": [], "motif_scrmsds": [], "clashes": []})
    per_model = defaultdict(lambda: {"n": 0, "succ": 0})
    per_motif = defaultdict(lambda: {"n": 0, "succ": 0})

    for r in rows:
        m = r["model_name"]; mo = r["motif_name"]
        succ = int(r["success"])
        per_mm[(m, mo)]["n"] += 1
        per_mm[(m, mo)]["succ"] += succ
        try:
            per_mm[(m, mo)]["scrmsds"].append(float(r["scrmsd"]))
            per_mm[(m, mo)]["motif_scrmsds"].append(float(r["motif_scrmsd"]))
        except ValueError:
            pass
        per_mm[(m, mo)]["clashes"].append(int(r["ligand_clashes"]))
        per_model[m]["n"] += 1
        per_model[m]["succ"] += succ
        per_motif[mo]["n"] += 1
        per_motif[mo]["succ"] += succ

    out_dir = args.out_dir
    import os; os.makedirs(out_dir, exist_ok=True)

    # Per (model, motif)
    with open(f"{out_dir}/by_model_motif.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "motif", "n", "successes", "pass_rate",
                    "mean_scrmsd", "mean_motif_scrmsd", "mean_clashes"])
        for (m, mo), v in sorted(per_mm.items()):
            mean_s = sum(v["scrmsds"]) / len(v["scrmsds"]) if v["scrmsds"] else float("nan")
            mean_ms = sum(v["motif_scrmsds"]) / len(v["motif_scrmsds"]) if v["motif_scrmsds"] else float("nan")
            mean_c = sum(v["clashes"]) / len(v["clashes"]) if v["clashes"] else 0
            rate = v["succ"] / v["n"] if v["n"] else 0
            w.writerow([m, mo, v["n"], v["succ"], f"{rate:.3f}",
                        f"{mean_s:.2f}", f"{mean_ms:.2f}", f"{mean_c:.2f}"])

    # Per model
    with open(f"{out_dir}/by_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "successes", "pass_rate"])
        for m, v in sorted(per_model.items()):
            rate = v["succ"] / v["n"] if v["n"] else 0
            w.writerow([m, v["n"], v["succ"], f"{rate:.3f}"])

    # Per motif
    with open(f"{out_dir}/by_motif.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motif", "n", "successes", "pass_rate"])
        for mo, v in sorted(per_motif.items()):
            rate = v["succ"] / v["n"] if v["n"] else 0
            w.writerow([mo, v["n"], v["succ"], f"{rate:.3f}"])

    # Print global summary
    n_total = sum(per_model[m]["n"] for m in per_model)
    n_succ = sum(per_model[m]["succ"] for m in per_model)
    print(f"\n=== Global ===")
    print(f"  total: {n_total}, successes: {n_succ}, pass rate: {100 * n_succ / max(1, n_total):.2f}%")
    print(f"\n=== By model ===")
    for m, v in sorted(per_model.items()):
        rate = v["succ"] / v["n"] if v["n"] else 0
        print(f"  {m:30s}  n={v['n']:6d}  succ={v['succ']:5d}  rate={100 * rate:.2f}%")
    print(f"\nwrote {out_dir}/{{by_model_motif,by_model,by_motif}}.csv")


if __name__ == "__main__":
    main()
