"""Build a compact cache of the PDB Chemical Component Dictionary (CCD).

For every chemical component we keep only what the atomized featurizer needs:

  * heavy-atom names (hydrogens/deuteriums dropped),
  * element symbols,
  * the *ideal* reference conformer coordinates (``pdbx_model_Cartn_*_ideal``,
    falling back to the experimental ``model_Cartn_*``), centred on the
    component centroid,
  * the component ``type`` string (used to tell polymer-linking components from
    non-polymers / saccharides),
  * the covalent bond list from ``_chem_comp_bond`` (indices into the heavy-atom
    name list, plus the bond order). Needed to write CONECT records, without
    which ligands render as disconnected spheres in every viewer.
  * ``mon_nstd_parent_comp_id`` -- the canonical residue a modified residue
    derives from. This is what lets a modified amino acid keep its parent's
    atom37 slots.

Output is a single pickle (~30 MB) so we do not pay 50k inodes for a directory
of per-component files.

Usage:
    python scripts/atomize/build_ccd_cache.py \
        --components /scratch/.../components.cif.gz \
        --out /scratch/.../ccd_cache.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import gemmi
import numpy as np


def build(components_path: str, out_path: str) -> None:
    t0 = time.time()
    doc = gemmi.cif.read(str(components_path))
    print(f"read {len(doc)} CCD blocks in {time.time() - t0:.1f}s", flush=True)

    cache: dict[str, dict] = {}
    for block in doc:
        code = block.name
        types = list(block.find_values("_chem_comp.type"))
        comp_type = types[0].strip('"').strip("'") if types else ""
        parents = list(block.find_values("_chem_comp.mon_nstd_parent_comp_id"))
        parent = parents[0].strip('"').strip("'") if parents else "?"
        if parent in ("?", ".", ""):
            parent = None

        table = block.find(
            "_chem_comp_atom.",
            [
                "atom_id",
                "type_symbol",
                "?pdbx_model_Cartn_x_ideal",
                "?pdbx_model_Cartn_y_ideal",
                "?pdbx_model_Cartn_z_ideal",
                "?model_Cartn_x",
                "?model_Cartn_y",
                "?model_Cartn_z",
            ],
        )
        names: list[str] = []
        elements: list[str] = []
        xyz: list[list[float]] = []
        for row in table:
            elem = row.str(1).strip().upper()
            if elem in ("H", "D", "T"):
                continue
            name = row.str(0).strip().strip('"')

            def _get(i: int) -> float:
                try:
                    if row.has(i):
                        v = row.str(i)
                        if v not in ("?", "."):
                            return float(v)
                except (ValueError, IndexError):
                    pass
                return np.nan

            pos = [_get(2), _get(3), _get(4)]
            if any(np.isnan(p) for p in pos):
                pos = [_get(5), _get(6), _get(7)]
            names.append(name)
            elements.append(elem)
            xyz.append(pos)

        if not names:
            continue

        # Covalent bonds, as indices into `names` (heavy atoms only).
        name_to_idx = {n: i for i, n in enumerate(names)}
        bond_table = block.find(
            "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "?value_order"]
        )
        bonds = []
        for row in bond_table:
            a1 = row.str(0).strip().strip('"')
            a2 = row.str(1).strip().strip('"')
            i1, i2 = name_to_idx.get(a1), name_to_idx.get(a2)
            if i1 is None or i2 is None:
                continue  # one partner is a hydrogen we dropped
            order = "SING"
            try:
                if row.has(2):
                    v = row.str(2).strip()
                    if v not in ("?", "."):
                        order = v
            except (ValueError, IndexError):
                pass
            bonds.append((i1, i2, order))
        arr = np.asarray(xyz, dtype=np.float32)
        # Missing ideal coords -> zeros (the featurizer emits a zeroed ref-conformer
        # channel for these, which is the same as "no reference available").
        bad = ~np.isfinite(arr).all(-1)
        if bad.any():
            arr[bad] = 0.0
        good = ~bad
        if good.any():
            arr = arr - arr[good].mean(0, keepdims=True)
            arr[bad] = 0.0
        else:
            arr[:] = 0.0

        cache[code] = {
            "names": names,
            "elements": elements,
            "ref_xyz": arr,
            "type": comp_type,
            "parent": parent,
            "has_ref": bool(good.any()),
            "bonds": bonds,
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(cache, fh, protocol=4)
    n_pep = sum(1 for v in cache.values() if "PEPTIDE" in v["type"].upper())
    n_par = sum(1 for v in cache.values() if v["parent"])
    n_bond = sum(1 for v in cache.values() if v["bonds"])
    print(
        f"wrote {len(cache)} components to {out_path} "
        f"({n_pep} peptide-linking, {n_par} with a canonical parent, "
        f"{n_bond} with bonds) in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--components",
        default="/scratch/users/tianyulu/datasets/pdb_atomized/components.cif.gz",
    )
    ap.add_argument(
        "--out",
        default="/scratch/users/tianyulu/datasets/pdb_atomized/ccd_cache.pkl",
    )
    args = ap.parse_args()
    build(args.components, args.out)
