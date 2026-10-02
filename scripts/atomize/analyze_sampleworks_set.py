"""What is actually in the Sampleworks 40-structure test set?

Reports, per entry, the modified amino acids and the non-polymer components the
atomized featurizer picks up, so we can see how much of the set needs ligand
handling rather than just non-canonical residue handling.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from protpardelle.common import residue_constants as rc
from protpardelle.data import atomize_io as A
from protpardelle.data.atomize_io import _is_peptide_linking, _parent_aatype

# The 40 entries, as listed by the Sampleworks group.
PDB_IDS = """8VPW 8VQ1 8Z76 9BN8 2I6H 2IGT 2E7Z 2P97 2QCV 3HYN 3I7M 3IAC 6RP1
3DT2 4GMU 7AVG 7P6M 5MHX 2XLW 2YL0 5I09 7Z0E 2BWD 7FRD 5RBR 2A26 5IMV 3HVV
3L4P 3T94 5A71 5MC8 6B8X 6DUR 6NI5 6NI6 6YVM 1VME 5SOP 4OLE""".split()

# Cryoprotectants, buffer and crystallisation additives: present in the file but
# not part of the biology. Kept separate from "real" ligands in the summary.
CRYSTALLISATION = {
    "HOH", "DOD", "SO4", "GOL", "EDO", "PO4", "CL", "NA", "K", "ACT", "MPD",
    "PEG", "TRS", "IMD", "DMS", "FMT", "NO3", "IOD", "BR", "UNX", "UNL",
    "PG4", "1PE", "EPE", "CIT", "ACY", "MES", "BME", "PGE", "2PE", "P6G",
    "TLA", "MLI", "NH4", "AZI", "SCN", "CAC", "BCT", "OXL", "MRD", "BU3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--max-tokens", type=int, default=100000, help="no crop by default")
    args = ap.parse_args()

    ccd = A.maybe_load_default_ccd()
    n_with_ncaa = n_with_ligand = n_with_metal = 0
    ligand_counter: collections.Counter = collections.Counter()
    ncaa_counter: collections.Counter = collections.Counter()
    rows = []
    missing = []

    for pdb_id in PDB_IDS:
        key = pdb_id.lower()
        path = Path(args.mirror) / key[1:3] / f"{key}.cif.gz"
        if not path.exists():
            missing.append(pdb_id)
            continue
        feats = A.load_feats_from_cif_atomized(
            str(path), ccd, max_tokens=args.max_tokens, crop=False,
            max_input_residues=100000, rng=np.random.default_rng(0),
        )
        if feats is None:
            missing.append(pdb_id + "(parse)")
            continue

        comps = feats["comp_names"]
        aat = feats["aatype"]
        ncaa, ligands, metals = set(), set(), set()
        for i, c in enumerate(comps):
            cu = c.strip().upper()
            if int(aat[i]) != A.UNK_AATYPE or _parent_aatype(ccd, cu) is not None and _is_peptide_linking(ccd, cu):
                if cu not in rc.restype_3to1 and cu != "UNK":
                    ncaa.add(cu)
                continue
            if cu in CRYSTALLISATION:
                continue
            entry = ccd.get(cu)
            n_heavy = len(entry["names"]) if entry else 0
            if n_heavy == 1:
                metals.add(cu)
            else:
                ligands.add(cu)
        ncaa_counter.update(ncaa)
        ligand_counter.update(ligands)
        n_with_ncaa += bool(ncaa)
        n_with_ligand += bool(ligands)
        n_with_metal += bool(metals)
        rows.append((pdb_id, feats["atom_positions"].shape[0], sorted(ncaa),
                     sorted(ligands), sorted(metals)))

    print(f"{'PDB':>6} {'tok':>5}  {'modified AA':<22} {'ligands':<34} metals")
    print("-" * 100)
    for pdb_id, ntok, ncaa, ligands, metals in rows:
        print(f"{pdb_id:>6} {ntok:5d}  {','.join(ncaa) or '-':<22} "
              f"{','.join(ligands) or '-':<34} {','.join(metals) or '-'}")

    n = len(rows)
    print(f"\n{n} entries parsed" + (f"; MISSING {missing}" if missing else ""))
    print(f"  with a modified amino acid : {n_with_ncaa:2d} / {n}")
    print(f"  with a non-polymer ligand  : {n_with_ligand:2d} / {n}")
    print(f"  with a metal ion           : {n_with_metal:2d} / {n}")
    print(f"  with ligand OR metal       : "
          f"{sum(1 for r in rows if r[3] or r[4]):2d} / {n}")
    print(f"  with none of the above     : "
          f"{sum(1 for r in rows if not (r[2] or r[3] or r[4])):2d} / {n}")
    print(f"\nmost common ligands: {ligand_counter.most_common(12)}")
    print(f"modified residues  : {ncaa_counter.most_common(12)}")


if __name__ == "__main__":
    main()
