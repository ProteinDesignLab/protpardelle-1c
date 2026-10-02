"""Select and balance the cc152 training set from the local wwPDB mirror.

Filters
-------
* entry type must contain protein (``prot`` / ``prot-nuc``),
* resolution must be present and <= ``--max-resolution`` (so NMR ensembles and
  unrefined models are excluded),
* between ``--min-res`` and ``--max-res`` observed residues,
* the mmCIF must actually be on disk in the mirror.

Balancing
---------
Entries are split into three buckets from the component index:

  ``ncaa``    contains at least one modified amino acid (a peptide-linking CCD
              component that is not one of the 20 canonical residues),
  ``ligand``  contains a non-polymer component of interesting size (or a metal),
  ``plain``   everything else.

``ncaa`` is the rare and valuable bucket, so it is oversampled by repeating its
keys until it reaches ``--ncaa-frac`` of the list; ``ligand`` likewise. The
validation list is held out by PDB id hash so it never overlaps training.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from protpardelle.common import residue_constants
from protpardelle.data import atomize_io

# NOT restype_3to1: that has 21 keys (it maps MSE -> M), which would class
# selenomethionine as canonical and keep MSE-only entries out of the
# oversampled NCAA bucket -- despite Se being the commonest extra-block atom.
CANONICAL = set(atomize_io.STANDARD_AA3)
# Components that are everywhere and say nothing interesting about an entry.
BORING = {
    "HOH", "DOD", "SO4", "GOL", "EDO", "PO4", "CL", "NA", "K", "MG", "CA",
    "ACT", "MPD", "PEG", "TRS", "IMD", "DMS", "FMT", "NO3", "IOD", "BR",
    "UNX", "UNL", "PG4", "1PE", "EPE", "CIT", "ACY", "MES", "BME",
}
METALS = {
    "ZN", "FE", "FE2", "CU", "CU1", "MN", "NI", "CO", "CD", "HG", "MO", "W",
    "V", "PT", "AU", "AG", "PB", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "YB", "LU", "LA", "CE", "PR", "ND", "Y1", "SR", "BA", "CS", "RB",
}


def load_entry_types(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2:
                out[f[0].lower()] = f[1]
    return out


def load_resolutions(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            pdb_id = f[0].strip().lower()
            if len(pdb_id) != 4:
                continue
            try:
                out[pdb_id] = float(f[6])
            except ValueError:
                continue
    return out


def classify(comps: dict[str, int], ccd: dict) -> str:
    has_ncaa = False
    has_ligand = False
    for comp, _count in comps.items():
        if comp in CANONICAL or comp in BORING:
            continue
        entry = ccd.get(comp)
        if entry is None:
            continue
        ctype = entry.get("type", "").upper()
        if "PEPTIDE LINKING" in ctype:
            has_ncaa = True
        elif comp in METALS:
            has_ligand = True
        elif "NON-POLYMER" in ctype or "SACCHARIDE" in ctype:
            n_heavy = len(entry.get("names", ()))
            if n_heavy >= 6:
                has_ligand = True
    if has_ncaa:
        return "ncaa"
    if has_ligand:
        return "ligand"
    return "plain"


def is_val(pdb_id: str, val_frac: float) -> bool:
    h = int(hashlib.md5(pdb_id.encode()).hexdigest()[:8], 16)
    return (h % 10000) < val_frac * 10000


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/scratch/users/tianyulu/datasets/pdb_atomized"
    ap.add_argument("--index-dir", default=f"{base}/index")
    ap.add_argument("--entry-types", default=f"{base}/pdb_entry_type.txt")
    ap.add_argument("--entries-idx", default=f"{base}/entries.idx")
    ap.add_argument("--ccd-cache", default=f"{base}/ccd_cache.pkl")
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--out-dir", default=base)
    ap.add_argument("--max-resolution", type=float, default=3.5)
    ap.add_argument("--min-res", type=int, default=30)
    ap.add_argument("--max-res", type=int, default=4000)
    ap.add_argument("--ncaa-frac", type=float, default=0.35)
    ap.add_argument("--ligand-frac", type=float, default=0.35)
    ap.add_argument("--val-frac", type=float, default=0.01)
    args = ap.parse_args()

    ccd = atomize_io.load_ccd_cache(args.ccd_cache)
    etypes = load_entry_types(args.entry_types)
    resolutions = load_resolutions(args.entries_idx)
    mirror = Path(args.mirror)

    buckets: dict[str, dict[str, list[str]]] = {
        "train": {"ncaa": [], "ligand": [], "plain": []},
        "val": {"ncaa": [], "ligand": [], "plain": []},
    }
    n_seen = n_kept = 0
    reasons = {"type": 0, "res": 0, "size": 0, "missing": 0}

    for tsv in sorted(Path(args.index_dir).glob("comps_*.tsv")):
        with open(tsv, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                pdb_id, _n_atoms, n_res, comp_str = parts[0], int(parts[1]), int(parts[2]), parts[3]
                n_seen += 1
                etype = etypes.get(pdb_id, "")
                if "prot" not in etype:
                    reasons["type"] += 1
                    continue
                resn = resolutions.get(pdb_id)
                if resn is None or resn > args.max_resolution:
                    reasons["res"] += 1
                    continue
                if not (args.min_res <= n_res <= args.max_res):
                    reasons["size"] += 1
                    continue
                if not (mirror / pdb_id[1:3] / f"{pdb_id}.cif.gz").exists():
                    reasons["missing"] += 1
                    continue
                comps = {}
                for item in comp_str.split(","):
                    if ":" in item:
                        k, v = item.rsplit(":", 1)
                        comps[k] = int(v)
                bucket = classify(comps, ccd)
                split = "val" if is_val(pdb_id, args.val_frac) else "train"
                buckets[split][bucket].append(pdb_id)
                n_kept += 1

    print(f"scanned {n_seen} indexed entries, kept {n_kept}; dropped {reasons}")
    for split in ("train", "val"):
        b = buckets[split]
        print(
            f"  {split}: ncaa={len(b['ncaa'])} ligand={len(b['ligand'])} "
            f"plain={len(b['plain'])}"
        )

    out_dir = Path(args.out_dir)
    for split in ("train", "val"):
        b = buckets[split]
        plain = b["plain"]
        keys: list[str] = list(plain)
        # Oversample the rare buckets to the requested fractions. Solving
        # n_x / (n_plain + n_x + n_y) = frac_x for the repeat counts.
        denom = max(1e-6, 1.0 - args.ncaa_frac - args.ligand_frac)
        for name, frac in (("ncaa", args.ncaa_frac), ("ligand", args.ligand_frac)):
            pool = b[name]
            if not pool:
                continue
            target = int(len(plain) * frac / denom)
            reps = max(1, round(target / len(pool)))
            keys.extend(pool * reps)
            print(f"  {split}/{name}: {len(pool)} unique x{reps} -> {len(pool) * reps}")
        keys.sort()
        path = out_dir / f"{split}_atomized_pdb_keys.list"
        with open(path, "w", encoding="utf-8") as fh:
            for k in keys:
                fh.write(k + "\n")
        print(f"wrote {len(keys)} keys to {path}")


if __name__ == "__main__":
    main()
