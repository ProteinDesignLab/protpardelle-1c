"""Sanity checks for the atomized featurizer on known non-canonical structures.

Prints, for each test entry, how the modified residue was laid out: which atoms
kept their canonical atom37 slot, which went into the global extra block, and
whether a CCD reference conformer was found.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from protpardelle.common import residue_constants as rc
from protpardelle.data import atomize_io as A

DEFAULT_ENTRIES = {
    "2xlw": "PCA",  # pyroglutamate
    "5imv": "OCS",  # cysteinesulfonic acid
    "6rp1": "KCX",  # carbamoylated lysine
    "6yvm": "ALY",  # N6-acetyl-lysine
    "5azd": "LYR",  # retinylidene-lysine (bacteriorhodopsin family)
    "7bmh": "LYR",
}


def describe(feats: dict, target: str) -> None:
    comps = feats["comp_names"]
    if target not in comps:
        print(f"   {target}: not in crop (comps: {sorted(set(comps))[:10]})")
        return
    i = comps.index(target)
    slots = np.nonzero(feats["atom_mask"][i].numpy())[0]
    canon = [s for s in slots if s < A.N_ATOM37]
    extra = [s for s in slots if s >= A.N_ATOM37]
    aa = feats["aatype"][i].item()
    aa_str = rc.restypes[aa] if aa < 20 else "X"
    names = feats["atom_names"][i]
    print(f"   {target}: aatype={aa_str} n_atoms={len(slots)}")
    print(
        f"      canonical slots  {canon}  -> "
        f"{[names[s] for s in canon]}"
    )
    print(
        f"      extra slots      {extra}  -> "
        f"{[names[s] for s in extra]} "
        f"elems={[A.ELEMENT_GROUPS[e] for e in feats['atom_elem'][i].numpy()[extra]]}"
    )
    n_ref = int((np.abs(feats["atom_ref"][i].numpy()[slots]).sum(-1) > 0).sum())
    print(f"      ref conformer present for {n_ref}/{len(slots)} atoms")
    assert all(feats["atom_noncanon"][i][s] == 0 for s in canon), "canonical slot flagged extra"
    assert all(feats["atom_noncanon"][i][s] == 1 for s in extra), "extra slot not flagged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args()

    ccd = A.maybe_load_default_ccd()
    for pdb_id, target in DEFAULT_ENTRIES.items():
        path = f"{args.mirror}/{pdb_id[1:3]}/{pdb_id}.cif.gz"
        t0 = time.time()
        feats = A.load_feats_from_cif_atomized(
            path, ccd, max_tokens=args.max_tokens, rng=np.random.default_rng(0)
        )
        if feats is None:
            print(f"{pdb_id}: FAILED TO LOAD ({path})")
            continue
        n = feats["atom_positions"].shape[0]
        n_extra = int(feats["atom_mask"][:, A.N_ATOM37 :].sum())
        print(
            f"{pdb_id}: tokens={n} atoms={int(feats['atom_mask'].sum())} "
            f"extra_block_atoms={n_extra} nmod={feats['n_modified']} "
            f"nhet={feats['n_hetero']} chains={int(feats['chain_index'].max()) + 1} "
            f"residx=[{feats['residue_index'].min():.0f},{feats['residue_index'].max():.0f}] "
            f"({time.time() - t0:.2f}s)"
        )
        # Nothing non-canonical may ever occupy a canonical slot.
        canon_flagged = feats["atom_noncanon"][:, : A.N_ATOM37].sum().item()
        het_rows = feats["aatype"] == A.UNK_AATYPE
        # ligand anchors are the one allowed exception (slot 1 copy)
        expected = float(het_rows.sum())
        assert canon_flagged <= expected, (
            f"{pdb_id}: {canon_flagged} canonical slots flagged extra, "
            f"only {expected} ligand anchors expected"
        )
        describe(feats, target)


if __name__ == "__main__":
    main()
