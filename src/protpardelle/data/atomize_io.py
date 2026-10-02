"""Atomized mmCIF loader: atom37 + non-canonical residues, ligands and metals.

Design goal: change as little of cc89 as possible.

cc89 represents a protein as ``[N_res, 37, 3]`` coordinates plus a ``[N_res, 37]``
atom mask, where slot ``k`` always means the atom named ``atom_types[k]``
("N", "CA", "C", "CB", "O", "CG", ...). Everything downstream -- the crop
conditioning, the noiser, the loss, MiniMPNN, the PDB writer -- is built on that
layout. So we keep it, and extend it in the smallest way that can express an
arbitrary biomolecular system:

1. **Modified amino acids keep their parent's atom37 slots.** The CCD records a
   ``mon_nstd_parent_comp_id`` for most modified residues (ALY->LYS, KCX->LYS,
   OCS->CYS, PCA->GLN, LYR->LYS, MSE->MET, ...). Every atom whose name is in the
   canonical atom37 namespace goes into its usual slot, and ``aatype`` is set to
   the parent residue. A modified Lys therefore looks *exactly* like a Lys to
   cc89, with extra atoms attached.

2. **Extra atoms go into a global extra-atom block, slots 37..60.** Crucially
   they do *not* reuse whichever atom37 slots the parent residue happens to
   leave free: that would put an acetyl carbon in "CG1"'s weights on a Lys and
   in some other residue's weights elsewhere, destroying exactly the weight
   sharing atom37 exists to provide. The extra block is the same for every
   residue type and every molecule, and atoms are ordered within it by
   (element group, CCD atom order), so the mapping from chemistry to slot is a
   property of the component alone. 24 extra slots cover 99.7% of
   modified-residue instances in the PDB.

3. **Ligands, cofactors and metals become pseudo-residue tokens** on their own
   chain index (so the 200-residue chain gap in the relative positional encoding
   separates them from the polymer). All of their atoms live in the extra block;
   the canonical slots stay empty apart from a copy of the anchor atom at slot 1,
   which keeps "slot 1 is this token's position" true everywhere downstream.
   Components with more than 24 atoms are split over several tokens.

4. **Three new per-atom input channels tell the trunk what each slot holds:**
   element one-hot (9), reference-conformer xyz from the CCD ideal coordinates
   (3), and a flag marking slots that are *not* being used with their canonical
   atom37 meaning (1). For a canonical protein these channels are fully
   determined by ``aatype``, so they carry no new information and the model
   behaves exactly like cc89 when their input weights are zero-initialised.

Nothing here changes the noise schedule, the loss, or the trunk.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import gemmi
import numpy as np
import torch

from protpardelle.common import residue_constants
from protpardelle.data.pdb_io import add_chain_gap

# --------------------------------------------------------------------------- #
# Element vocabulary for the per-atom element channel.
# --------------------------------------------------------------------------- #
ELEMENT_GROUPS = ("C", "N", "O", "S", "P", "SE", "HALOGEN", "METAL", "OTHER")
N_ELEMENT_GROUPS = len(ELEMENT_GROUPS)
_HALOGENS = {"F", "CL", "BR", "I", "AT"}
_NONMETALS = {
    "H", "D", "C", "N", "O", "S", "P", "SE", "B", "SI", "AS", "TE",
    "F", "CL", "BR", "I", "AT", "HE", "NE", "AR", "KR", "XE", "RN",
}

# Per-atom reference-conformer coordinates are divided by this before being fed
# to the network, so they land in roughly the same numeric range as the
# sigma_data-normalised input coordinates.
REF_COORD_SCALE = 10.0

# Bond-topology channel scales (cc154). Bond lengths live in ~1.2-2.1 A and
# anchor distances in ~0-10 A; dividing keeps both near unity.
BOND_LEN_SCALE = 2.0
ANCHOR_DIST_SCALE = 10.0
MAX_BOND_DEGREE = 4.0
# Per-example cap on the bonded pairs handed to the loss. 40 Sampleworks
# structures average ~20 non-standard bonds each, so this is generous.
MAX_BOND_PAIRS = 512

N_ATOM37 = residue_constants.atom_type_num  # 37
UNK_AATYPE = residue_constants.restype_num  # 20

# --------------------------------------------------------------------------- #
# Slot layout.
#
# Slots 0..36 are the untouched atom37 namespace: slot k *always* means the atom
# named ``atom_types[k]``. This is the inductive bias that makes atom37 work --
# the CB of an Ala and the CB of a Trp hit the same first-layer weights -- and
# nothing non-canonical is ever allowed into these slots.
#
# Slots 37..37+N_EXTRA_SLOTS-1 are a *global* extra-atom block with exactly the
# same property one level up: extra slot j means "the j-th extra atom of this
# component", with the same weights for every residue type and every molecule.
# The acetyl carbon of an N-acetyl-lysine and the acetyl carbon of an acetylated
# serine therefore land in the same slot and share a representation, which they
# would not if extra atoms reused whichever atom37 slots the parent happens to
# leave free.
#
# Extra atoms are ordered deterministically by (element group, CCD atom order),
# so the ordering is a property of the chemistry, not of the parent residue or
# of the order atoms happen to appear in a given PDB file.
#
# N_EXTRA_SLOTS = 24 covers 99.7% of modified-residue instances in the PDB
# (retinylidene-lysine needs 21, PLP-lysine 15; most PTMs need 1-4).
# --------------------------------------------------------------------------- #
N_EXTRA_SLOTS = 24
N_SLOTS = N_ATOM37 + N_EXTRA_SLOTS  # 61
EXTRA_SLOTS = list(range(N_ATOM37, N_SLOTS))

# Ligands/metals have no canonical atom names at all, so every one of their
# atoms lives in the extra block. Slot 1 additionally carries a *copy* of the
# token's anchor atom: throughout the codebase slot 1 ("CA") is the per-token
# position used for neighbour and crop-conditioning distances, so keeping it
# populated means ligand tokens need no special-casing anywhere downstream.
ANCHOR_SLOT = residue_constants.atom_order["CA"]

# The 20 standard residues, explicitly. NOT `residue_constants.restype_3to1`,
# which has 21 keys because protpardelle maps MSE (selenomethionine) to M.
# MSE is a *modified* residue here: its SE atom has no atom37 name and goes into
# the extra block, so treating it as standard silently denied it bond records,
# CONECT output, and oversampling in the training mix -- while Se is the single
# most common extra-block atom in the PDB.
STANDARD_AA3 = frozenset(
    residue_constants.restype_1to3[a] for a in residue_constants.restypes[:20]
)
MAX_LIGAND_ATOMS_PER_TOKEN = N_EXTRA_SLOTS  # 24


def element_group_index(symbol: str) -> int:
    """Map an element symbol onto the small element vocabulary."""

    s = symbol.strip().upper()
    if s in ("C", "N", "O", "S", "P", "SE"):
        return ELEMENT_GROUPS.index(s)
    if s in _HALOGENS:
        return ELEMENT_GROUPS.index("HALOGEN")
    if s and s not in _NONMETALS:
        return ELEMENT_GROUPS.index("METAL")
    return ELEMENT_GROUPS.index("OTHER")


# --------------------------------------------------------------------------- #
# CCD cache
# --------------------------------------------------------------------------- #
_CCD_CACHE: dict[str, dict] | None = None
_CCD_PATH_LOADED: str | None = None


def load_ccd_cache(path: str) -> dict[str, dict]:
    """Load (and memoise per process) the pickle written by build_ccd_cache.py."""

    global _CCD_CACHE, _CCD_PATH_LOADED
    if _CCD_CACHE is not None and _CCD_PATH_LOADED == str(path):
        return _CCD_CACHE
    with open(path, "rb") as fh:
        _CCD_CACHE = pickle.load(fh)
    _CCD_PATH_LOADED = str(path)
    return _CCD_CACHE


# Canonical atom37 slot occupancy per aatype, built once.
def _canonical_slot_mask() -> np.ndarray:
    """[21, 37] bool: which atom37 slots each canonical residue type uses."""

    mask = np.zeros((UNK_AATYPE + 1, N_ATOM37), dtype=bool)  # noqa: E501
    for aa_idx in range(UNK_AATYPE):
        aa1 = residue_constants.restypes[aa_idx]
        aa3 = residue_constants.restype_1to3[aa1]
        for name in residue_constants.residue_atoms[aa3]:
            mask[aa_idx, residue_constants.atom_order[name]] = True
    # UNK uses backbone only.
    for name in ("N", "CA", "C", "O"):
        mask[UNK_AATYPE, residue_constants.atom_order[name]] = True
    return mask


CANONICAL_SLOT_MASK = _canonical_slot_mask()


# --------------------------------------------------------------------------- #
# Token building
# --------------------------------------------------------------------------- #
class _Token:
    """One row of the atom37 tensors."""

    __slots__ = (
        "pos", "mask", "elem", "ref", "noncanon", "aatype",
        "residue_index", "chain_index", "bfac", "comp", "atom_names", "inst",
        "bondfeat",
    )

    def __init__(self):
        self.pos = np.zeros((N_SLOTS, 3), dtype=np.float32)
        self.mask = np.zeros((N_SLOTS,), dtype=np.float32)
        self.elem = np.zeros((N_SLOTS,), dtype=np.int64)
        self.ref = np.zeros((N_SLOTS, 3), dtype=np.float32)
        self.noncanon = np.zeros((N_SLOTS,), dtype=np.float32)
        self.bfac = np.zeros((N_SLOTS,), dtype=np.float32)
        self.aatype = UNK_AATYPE
        self.residue_index = 0
        self.chain_index = 0
        self.comp = ""
        self.atom_names: dict[int, str] = {}
        self.inst = -1  # residue-instance id; shared by the tokens of a split ligand
        # [N_SLOTS, 3]: (d_parent, d_anchor, n_bonds), scaled. cc154.
        self.bondfeat = np.zeros((N_SLOTS, 3), dtype=np.float32)


def _residue_atoms(res) -> list:
    """Heavy atoms of a gemmi residue at the first altloc, deduplicated by name."""

    seen: set[str] = set()
    out = []
    for atom in res:
        if atom.element.is_hydrogen:
            continue
        if atom.altloc not in ("\x00", "", "A"):
            continue
        name = atom.name.strip()
        if name in seen:
            continue
        seen.add(name)
        out.append(atom)
    return out


def _ref_lookup(ccd: dict, comp: str) -> dict[str, np.ndarray]:
    entry = ccd.get(comp)
    if entry is None or not entry.get("has_ref", False):
        return {}
    return {n: xyz for n, xyz in zip(entry["names"], entry["ref_xyz"])}


def _parent_aatype(ccd: dict, comp: str) -> int | None:
    """Canonical aatype a (possibly modified) residue derives from, else None."""

    if comp in residue_constants.restype_3to1:
        return residue_constants.restype_order[residue_constants.restype_3to1[comp]]
    entry = ccd.get(comp)
    if entry is None:
        return None
    parent = entry.get("parent")
    if parent and parent in residue_constants.restype_3to1:
        return residue_constants.restype_order[residue_constants.restype_3to1[parent]]
    return None


def _is_peptide_linking(ccd: dict, comp: str) -> bool:
    entry = ccd.get(comp)
    if entry is None:
        return False
    return "PEPTIDE LINKING" in entry.get("type", "").upper()


def component_bond_features(entry: dict, anchor_name: str | None):
    """Rotation-invariant bond-topology scalars for one CCD component.

    Returns ``{atom_name: (d_parent, d_anchor, n_bonds)}`` in Angstrom, where
    ``d_parent`` is the ideal distance to the atom's parent in a BFS spanning
    tree of the bond graph rooted at ``anchor_name``.

    These are fed instead of raw reference coordinates because the network has
    no way to compute a distance between two slots of the same token: the patch
    embedding is a single linear map over the flattened slots and the
    transformer attends between tokens, so ``|ref_a - ref_b|`` is never formed.
    Supplying the distance directly is what the model actually needs.
    """

    names = entry.get("names") or []
    bonds = entry.get("bonds") or []
    ref = entry.get("ref_xyz")
    if not names or not bonds or ref is None:
        return {}

    idx = {n: i for i, n in enumerate(names)}
    adj: dict[int, list[int]] = {i: [] for i in range(len(names))}
    for i1, i2, _o in bonds:
        if i1 < len(names) and i2 < len(names):
            adj[i1].append(i2)
            adj[i2].append(i1)

    root = idx.get(anchor_name) if anchor_name else None
    if root is None:
        # fall back to the most-connected atom, a stable choice for a ligand
        root = max(range(len(names)), key=lambda i: len(adj[i]))

    parent = {root: None}
    order = [root]
    while order:
        cur = order.pop(0)
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                order.append(nb)

    def dist(i, j):
        return float(np.linalg.norm(ref[i] - ref[j]))

    out = {}
    for i, nm in enumerate(names):
        pa = parent.get(i)
        d_par = dist(i, pa) if pa is not None else 0.0
        d_anc = dist(i, root)
        out[nm] = (d_par, d_anc, float(len(adj[i])))
    return out


def _place(tok: _Token, slot: int, atom, refs: dict, is_extra: bool) -> None:
    name = atom.name.strip()
    tok.pos[slot] = [atom.pos.x, atom.pos.y, atom.pos.z]
    tok.mask[slot] = 1.0
    tok.bfac[slot] = atom.b_iso
    tok.elem[slot] = element_group_index(atom.element.name)
    tok.noncanon[slot] = 1.0 if is_extra else 0.0
    tok.atom_names[slot] = name
    r = refs.get(name)
    if r is not None:
        tok.ref[slot] = r


def _extra_sort_key(ccd_order: dict[str, int]):
    """Deterministic, chemistry-driven ordering for the extra-atom block.

    Sorting by element group first means the extra block reads carbons, then
    nitrogens, then oxygens, ... for every component in the PDB, so a given
    extra slot has a consistent meaning across molecules. CCD atom order breaks
    ties, which makes the assignment a property of the component rather than of
    the order atoms appear in a particular deposition.
    """

    def key(atom):
        name = atom.name.strip()
        return (
            element_group_index(atom.element.name),
            ccd_order.get(name, 10_000),
            name,
        )

    return key


def _fill_bondfeat(tok: _Token, ccd: dict, anchor_name: str | None) -> None:
    """Populate the token's bond-topology channels from the CCD bond graph."""

    entry = ccd.get(tok.comp.strip().upper())
    if entry is None:
        return
    bf = component_bond_features(entry, anchor_name)
    if not bf:
        return
    for slot, nm in tok.atom_names.items():
        v = bf.get(nm)
        if v is None:
            continue
        tok.bondfeat[slot] = (
            v[0] / BOND_LEN_SCALE,
            v[1] / ANCHOR_DIST_SCALE,
            v[2] / MAX_BOND_DEGREE,
        )


def _make_polymer_token(res, ccd: dict, refs: dict, aatype: int) -> _Token | None:
    """A canonical or modified amino-acid token.

    Atoms whose names are in the atom37 namespace keep their atom37 slot -- this
    is what makes a modified Lys look like a Lys to cc89. Everything else goes
    into the global extra block, never into a canonical slot.
    """

    tok = _Token()
    tok.aatype = aatype
    tok.comp = res.name
    entry = ccd.get(res.name.strip().upper())
    ccd_order = (
        {n: i for i, n in enumerate(entry["names"])} if entry is not None else {}
    )

    extras = []
    for atom in _residue_atoms(res):
        name = atom.name.strip()
        slot = residue_constants.atom_order.get(name)
        if slot is not None and tok.mask[slot] == 0:
            _place(tok, slot, atom, refs, is_extra=False)
        else:
            extras.append(atom)

    for slot, atom in zip(EXTRA_SLOTS, sorted(extras, key=_extra_sort_key(ccd_order))):
        _place(tok, slot, atom, refs, is_extra=True)

    if tok.mask.sum() < 0.5:
        return None
    _fill_bondfeat(tok, ccd, "CA")
    return tok


def _make_hetero_tokens(res, ccd: dict, refs: dict) -> list[_Token]:
    """One or more pseudo-residue tokens for a ligand / cofactor / metal.

    Ligand atoms have no canonical atom names, so they live entirely in the
    extra block; the canonical atom37 slots stay empty apart from a copy of the
    anchor atom at slot 1 (see ANCHOR_SLOT).
    """

    atoms = _residue_atoms(res)
    if not atoms:
        return []
    entry = ccd.get(res.name.strip().upper())
    ccd_order = (
        {n: i for i, n in enumerate(entry["names"])} if entry is not None else {}
    )
    atoms = sorted(atoms, key=_extra_sort_key(ccd_order))

    tokens = []
    for start in range(0, len(atoms), MAX_LIGAND_ATOMS_PER_TOKEN):
        chunk = atoms[start : start + MAX_LIGAND_ATOMS_PER_TOKEN]
        tok = _Token()
        tok.aatype = UNK_AATYPE
        tok.comp = res.name
        for slot, atom in zip(EXTRA_SLOTS, chunk):
            _place(tok, slot, atom, refs, is_extra=True)
        # Anchor: the atom of this chunk nearest its centroid, duplicated into
        # slot 1 so every token has a well-defined position.
        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in chunk], dtype=np.float32)
        anchor = chunk[int(np.argmin(np.linalg.norm(xyz - xyz.mean(0), axis=1)))]
        _place(tok, ANCHOR_SLOT, anchor, refs, is_extra=True)
        _fill_bondfeat(tok, ccd, anchor.name.strip())
        tokens.append(tok)
    return tokens


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def load_feats_from_cif_atomized(
    path: str,
    ccd: dict,
    *,
    chain_residx_gap: int = 200,
    max_tokens: int = 384,
    max_input_residues: int = 6000,
    contact_cutoff: float = 10.0,
    include_hetero: bool = True,
    min_hetero_atoms: int = 1,
    anchor_on_hetero_prob: float = 0.8,
    rng: np.random.Generator | None = None,
    crop: bool = True,
) -> dict | None:
    """Featurize an mmCIF (typically a bioassembly) into atom37 + atomized feats.

    Returns None if the file yields nothing usable. Large assemblies are cropped
    to ``max_tokens`` tokens around an anchor, preferentially a non-canonical
    residue / ligand so those are actually seen during training.
    """

    rng = rng or np.random.default_rng()
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        return None
    if len(st) == 0:
        return None
    st.setup_entities()
    model = st[0]

    n_res_total = sum(len(ch) for ch in model)
    if n_res_total == 0 or n_res_total > max_input_residues:
        return None

    inst_counter = 0                 # unique id per residue instance
    poly_tokens: list[_Token] = []   # polymer residues, in chain/sequence order
    het_tokens: list[_Token] = []    # ligands / metals / cofactors
    chain_of_poly: list[str] = []
    n_modified = 0

    for chain in model:
        for res in chain:
            comp = res.name.strip().upper()
            info = gemmi.find_tabulated_residue(comp)
            if info is not None and info.is_water():
                continue
            refs = _ref_lookup(ccd, comp)
            aatype = _parent_aatype(ccd, comp)
            is_poly_aa = aatype is not None and (
                comp in residue_constants.restype_3to1 or _is_peptide_linking(ccd, comp)
            )
            if is_poly_aa:
                # Require a backbone to count as a polymer residue.
                names = {a.name.strip() for a in _residue_atoms(res)}
                if not {"N", "CA", "C"} <= names:
                    is_poly_aa = False
            if is_poly_aa:
                tok = _make_polymer_token(res, ccd, refs, aatype)
                if tok is None:
                    continue
                tok.residue_index = res.seqid.num
                tok.inst = inst_counter
                inst_counter += 1
                poly_tokens.append(tok)
                chain_of_poly.append(chain.name)
                if comp not in STANDARD_AA3:
                    n_modified += 1
            elif include_hetero:
                if len(_residue_atoms(res)) < min_hetero_atoms:
                    continue
                new = _make_hetero_tokens(res, ccd, refs)
                for t in new:
                    t.residue_index = res.seqid.num
                    t.inst = inst_counter
                inst_counter += 1
                het_tokens.extend(new)

    if len(poly_tokens) < 5:
        return None

    poly_xyz = np.stack([t.pos[1] for t in poly_tokens])  # CA positions

    # ----------------------------------------------------------------- crop
    keep_poly = list(range(len(poly_tokens)))
    if crop and len(poly_tokens) > max_tokens:
        het_anchor = None
        if het_tokens and rng.random() < anchor_on_hetero_prob:
            het_anchor = het_tokens[rng.integers(len(het_tokens))]
        elif n_modified and rng.random() < anchor_on_hetero_prob:
            mod_idx = [
                i for i, t in enumerate(poly_tokens)
                if t.comp.strip().upper() not in STANDARD_AA3
            ]
            het_anchor = poly_tokens[mod_idx[rng.integers(len(mod_idx))]]
        if het_anchor is not None:
            centre = het_anchor.pos[het_anchor.mask > 0].mean(0)
        else:
            centre = poly_xyz[rng.integers(len(poly_xyz))]
        order = np.argsort(np.linalg.norm(poly_xyz - centre, axis=1))
        budget = max(max_tokens - min(len(het_tokens), max_tokens // 4), 16)
        keep_poly = sorted(order[:budget].tolist())

    keep_set = set(keep_poly)
    kept_poly = [poly_tokens[i] for i in keep_poly]
    kept_chains = [chain_of_poly[i] for i in keep_poly]

    # Keep hetero tokens in contact with the kept polymer.
    kept_het: list[_Token] = []
    if het_tokens and kept_poly:
        kept_xyz = np.stack([poly_xyz[i] for i in keep_poly])
        for t in het_tokens:
            hx = t.pos[t.mask > 0]
            if hx.size == 0:
                continue
            d = np.linalg.norm(kept_xyz[:, None, :] - hx[None, :, :], axis=-1).min()
            if d <= contact_cutoff:
                kept_het.append(t)
        kept_het = kept_het[: max(max_tokens - len(kept_poly), 0)]

    tokens = kept_poly + kept_het
    if len(tokens) < 5:
        return None

    # ------------------------------------------------------- chain indices
    # Polymer chains keep their identity; every hetero component instance gets
    # its own chain index so `add_chain_gap` puts a 200-residue break between it
    # and everything else (no spurious sequence adjacency).
    chain_order: list[str] = []
    for c in kept_chains:
        if c not in chain_order:
            chain_order.append(c)
    chain_map = {c: i for i, c in enumerate(chain_order)}
    chain_index = [chain_map[c] for c in kept_chains]
    next_chain = len(chain_order)
    for i, t in enumerate(kept_het):
        chain_index.append(next_chain + i)
        t.residue_index = 1

    feats = {
        "atom_positions": torch.from_numpy(np.stack([t.pos for t in tokens])).float(),
        "atom_mask": torch.from_numpy(np.stack([t.mask for t in tokens])).float(),
        "b_factors": torch.from_numpy(np.stack([t.bfac for t in tokens])).float(),
        "aatype": torch.tensor([t.aatype for t in tokens], dtype=torch.long),
        "residue_index": torch.tensor(
            [t.residue_index for t in tokens], dtype=torch.float32
        ),
        "chain_index": torch.tensor(chain_index, dtype=torch.float32),
        "atom_elem": torch.from_numpy(np.stack([t.elem for t in tokens])).long(),
        "atom_ref": torch.from_numpy(np.stack([t.ref for t in tokens])).float()
        / REF_COORD_SCALE,
        "atom_noncanon": torch.from_numpy(
            np.stack([t.noncanon for t in tokens])
        ).float(),
    }
    # Explicit ligand/metal row flag. `aatype == UNK` is not sufficient: a real
    # UNK residue in the polymer also carries aatype 20.
    is_lig = np.zeros(len(tokens), dtype=np.float32)
    is_lig[len(kept_poly):] = 1.0
    feats["is_ligand"] = torch.from_numpy(is_lig)

    # ------------------------------------------------------------- bonds
    # Covalent bonds from the CCD, expressed as (token_i, slot_i, token_j,
    # slot_j) so that a component split across several tokens still carries its
    # cross-token bonds. Only non-standard components are recorded: viewers
    # build standard amino acids from their own templates, and that is also what
    # a PDB converted from the source mmCIF contains. Without these, every
    # ligand renders as a cloud of disconnected spheres.
    slot_of: dict[tuple[int, str], int] = {}
    for ti, t in enumerate(tokens):
        for slot, nm in t.atom_names.items():
            if t.mask[slot] > 0 and not (t.inst >= 0 and slot == ANCHOR_SLOT
                                         and t.aatype == UNK_AATYPE):
                slot_of.setdefault((t.inst, nm), (ti, slot))
    token_of_inst: dict[int, list[int]] = {}
    for ti, t in enumerate(tokens):
        token_of_inst.setdefault(t.inst, []).append(ti)

    bond_list = []
    seen_inst = set()
    for ti, t in enumerate(tokens):
        comp = t.comp.strip().upper()
        if t.inst in seen_inst:
            continue
        seen_inst.add(t.inst)
        if comp in STANDARD_AA3:
            continue  # standard residue: viewers have a template
        entry = ccd.get(comp)
        if not entry or not entry.get("bonds"):
            continue
        cnames = entry["names"]
        for i1, i2, _order in entry["bonds"]:
            if i1 >= len(cnames) or i2 >= len(cnames):
                continue
            a = slot_of.get((t.inst, cnames[i1]))
            b = slot_of.get((t.inst, cnames[i2]))
            if a is not None and b is not None:
                bond_list.append((a[0], a[1], b[0], b[1]))
    feats["bond_list"] = bond_list
    feats["atom_bondfeat"] = torch.from_numpy(
        np.stack([t.bondfeat for t in tokens])
    ).float()

    # Padded (token_i, slot_i, token_j, slot_j) pairs for the bonded-distance
    # loss, with the ideal length in a parallel tensor. -1 marks padding.
    pairs = torch.full((MAX_BOND_PAIRS, 4), -1, dtype=torch.long)
    plen = torch.zeros(MAX_BOND_PAIRS, dtype=torch.float32)
    ref_all = feats["atom_ref"].numpy() * REF_COORD_SCALE
    n_kept = 0
    for ti, si, tj, sj in bond_list:
        if n_kept >= MAX_BOND_PAIRS:
            break
        d = float(np.linalg.norm(ref_all[ti, si] - ref_all[tj, sj]))
        if not (0.5 < d < 3.0):
            continue  # no usable ideal geometry for this pair
        pairs[n_kept] = torch.tensor([ti, si, tj, sj])
        plen[n_kept] = d
        n_kept += 1
    feats["bond_pairs"] = pairs
    feats["bond_lengths"] = plen

    feats["bb_coords"] = feats["atom_positions"][
        :, [residue_constants.atom_order[a] for a in ("N", "CA", "C", "O")]
    ].clone()

    # Residue indices: make them monotone within the crop, then apply the
    # standard 200-residue inter-chain gap used by the relative positional
    # encoding. Gaps inside a chain (from cropping / missing density) are kept.
    ri = feats["residue_index"]
    feats["residue_index"] = ri - ri.min() + 1
    feats["residue_index"] = add_chain_gap(
        feats["residue_index"], feats["chain_index"], chain_residx_gap=chain_residx_gap
    )

    feats["n_modified"] = int(
        sum(1 for t in kept_poly if t.comp.strip().upper() not in STANDARD_AA3)
    )
    feats["n_hetero"] = len(kept_het)
    feats["comp_names"] = [t.comp for t in tokens]
    feats["atom_names"] = [t.atom_names for t in tokens]
    return feats


def feats_to_pdb_str_atomized(
    feats: dict,
    coords: "torch.Tensor | np.ndarray | None" = None,
    b_factors: "torch.Tensor | np.ndarray | None" = None,
) -> str:
    """Write atomized features back out as a PDB string.

    The featurizer keeps the real CCD atom name for every occupied slot in
    ``feats["atom_names"]``, so the extra-block atoms come back out with their
    proper names (``C15`` of a retinal, ``OQ1`` of a carbamate, ...) rather than
    as whatever atom37 slot they happened to use. Ligand anchor duplicates at
    slot 1 are skipped so no atom is written twice.

    CONECT records are emitted for every bond in ``feats["bond_list"]`` (CCD
    connectivity for non-standard components), so ligands and modified-residue
    decorations render bonded rather than as isolated spheres.

    Args:
        feats: output of :func:`load_feats_from_cif_atomized` (needs the
            non-tensor ``comp_names`` / ``atom_names`` / ``bond_list``
            bookkeeping, so call it directly rather than going through the
            dataset).
        coords: [N, N_SLOTS, 3] coordinates to write; defaults to the input
            coordinates. Pass a model prediction here to write a sample.
    """

    import numpy as _np

    xyz = feats["atom_positions"] if coords is None else coords
    xyz = _np.asarray(xyz.detach().cpu() if hasattr(xyz, "detach") else xyz)
    mask = _np.asarray(feats["atom_mask"].detach().cpu())
    bfac = feats.get("b_factors") if b_factors is None else b_factors
    bfac = _np.asarray(bfac.detach().cpu()) if bfac is not None else _np.zeros_like(mask)
    aatype = _np.asarray(feats["aatype"].detach().cpu())
    res_idx = _np.asarray(feats["residue_index"].detach().cpu())
    chain_idx = _np.asarray(feats["chain_index"].detach().cpu())
    comp_names = feats["comp_names"]
    atom_names = feats["atom_names"]

    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    lines = []
    serial_of: dict[tuple[int, int], int] = {}
    serial = 1
    for i in range(xyz.shape[0]):
        is_hetero = int(aatype[i]) == UNK_AATYPE
        resname = comp_names[i][:3].rjust(3)
        ch = chain_letters[int(chain_idx[i]) % len(chain_letters)]
        resseq = int(res_idx[i]) % 10000
        names_i = atom_names[i]
        anchor_name = names_i.get(ANCHOR_SLOT) if is_hetero else None
        for slot in range(xyz.shape[1]):
            if mask[i, slot] <= 0:
                continue
            name = names_i.get(slot)
            if name is None:
                continue
            # A ligand's anchor copy at slot 1 duplicates an extra-block atom.
            if is_hetero and slot == ANCHOR_SLOT and anchor_name is not None:
                continue
            elem = "".join(c for c in name if c.isalpha())[:1] or "C"
            pdb_name = f" {name:<3s}" if len(name) < 4 else name
            x, y, z = xyz[i, slot]
            record = "HETATM" if is_hetero else "ATOM  "
            lines.append(
                f"{record}{serial:5d} {pdb_name:<4s}{resname:>4s} {ch}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{float(bfac[i, slot]):6.2f}"
                f"          {elem:>2s}"
            )
            serial_of[(i, slot)] = serial
            serial += 1

    # CONECT records. Without these, every ligand and every modified-residue
    # decoration renders as a cloud of disconnected spheres: viewers build
    # standard amino acids from templates but have nothing to go on for a
    # HETATM or a non-standard component. Bonds come from the CCD
    # (`_chem_comp_bond`), so they are the deposited connectivity, not distance
    # guesses. Matches what a PDB converted from the source mmCIF contains:
    # CONECT for non-standard components only.
    partners: dict[int, list[int]] = {}
    for ti, si, tj, sj in feats.get("bond_list", []):
        a, b = serial_of.get((ti, si)), serial_of.get((tj, sj))
        if a is None or b is None:
            continue
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    for a in sorted(partners):
        ps = sorted(set(partners[a]))
        # PDB allows at most 4 partners per CONECT line; continue on more lines.
        for k in range(0, len(ps), 4):
            chunk = ps[k : k + 4]
            lines.append(
                f"CONECT{a:5d}" + "".join(f"{b:5d}" for b in chunk)
            )

    lines.append("END")
    return "\n".join(lines) + "\n"


def default_ccd_path() -> str:
    import os

    return os.environ.get(
        "PROTPARDELLE_CCD_CACHE",
        "/scratch/users/tianyulu/datasets/pdb_atomized/ccd_cache.pkl",
    )


def maybe_load_default_ccd() -> dict:
    p = default_ccd_path()
    if not Path(p).exists():
        raise FileNotFoundError(
            f"CCD cache not found at {p}; run scripts/atomize/build_ccd_cache.py"
        )
    return load_ccd_cache(p)
