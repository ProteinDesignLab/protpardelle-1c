from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB import (
    PDBIO,
    Atom,
    Chain,
    Model,
    PDBParser,
    Polypeptide,
    Residue,
    Structure,
)

# ---------- small helpers ----------


def _new_atom(
    name: str,
    coord,
    bfactor: float = 20.0,
    occupancy: float = 1.0,
    element: Optional[str] = None,
    serial_number: int = 0,
) -> Atom.Atom:
    """Create a new Biopython Atom with sensible defaults."""
    fullname = name.rjust(4)  # PDB columns require right-justified 4 chars
    if element is None:
        e = name.strip()[:1] or "C"
        element = e if e.isalpha() else "C"
    return Atom.Atom(
        name=name,
        coord=np.array(coord, dtype=float),
        bfactor=float(bfactor),
        occupancy=float(occupancy),
        altloc=" ",
        fullname=fullname,
        serial_number=int(serial_number),
        element=element,
    )


def _copy_atom_with_new_name(
    src: Atom.Atom, new_name: str, new_element: Optional[str] = None
) -> Atom.Atom:
    return _new_atom(
        new_name,
        src.get_coord(),
        bfactor=src.get_bfactor(),
        occupancy=src.get_occupancy() if src.get_occupancy() is not None else 1.0,
        element=(new_element if new_element is not None else src.element),
    )


def _new_residue(
    resname: str, resseq: int, hetero: bool = False, icode: str = " "
) -> Residue.Residue:
    hetfield = "H_" + resname if hetero else " "
    return Residue.Residue((hetfield, int(resseq), icode), resname, segid="    ")


def _is_dtr(res: Residue.Residue) -> bool:
    return res.get_resname().strip().upper() == "DTR" and str(res.id[0]).startswith(
        "H_"
    )


def _is_cys(res: Residue.Residue) -> bool:
    return res.get_resname().strip().upper() == "CYS" and res.id[0] == " "


# ---------- forward: thioether -> native ----------


def _build_chain_native_from_thioether(chain: Chain.Chain) -> Tuple[Chain.Chain, bool]:
    """
    For chains that match: first residue HETATM DTR and last residue CYS,
    build a native chain: GLY1 (SG->N, CP2->CA, CO->C, OP1->O), TRP2 (DTR minus those three),
    middle residues shifted by +1, and final CYS->ALA (drop SG).
    """
    residues = list(chain.get_residues())
    if not residues:
        return chain.copy(), False

    first, last = residues[0], residues[-1]
    if not (_is_dtr(first) and _is_cys(last)):
        # Not a target chain: return a renumbered copy
        new_ch = Chain.Chain(chain.id)
        for i, res in enumerate(residues, start=1):
            het = str(res.id[0]).startswith("H_")
            new_res = _new_residue(res.get_resname(), i, hetero=het)
            for at in res.get_atoms():
                new_res.add(
                    _new_atom(
                        at.get_name(),
                        at.get_coord(),
                        at.get_bfactor(),
                        at.get_occupancy() or 1.0,
                        at.element,
                    )
                )
            new_ch.add(new_res)
        return new_ch, False

    dtr = first
    cys = last
    if not {"CP2", "CO", "OP1"}.issubset(
        {a.get_name().upper() for a in dtr.get_atoms()}
    ):
        raise ValueError(f"Chain {chain.id}: DTR must have CP2/CO/OP1.")
    if "SG" not in {a.get_name().upper() for a in cys.get_atoms()}:
        raise ValueError(f"Chain {chain.id}: CYS must have SG.")

    # GLY1 from CYS.SG and DTR.{CP2, CO, OP1}
    gly1 = _new_residue("GLY", 1, hetero=False)
    gly1.add(_copy_atom_with_new_name(cys["SG"], "N", "N"))
    gly1.add(_copy_atom_with_new_name(dtr["CP2"], "CA", "C"))
    gly1.add(_copy_atom_with_new_name(dtr["CO"], "C", "C"))
    gly1.add(_copy_atom_with_new_name(dtr["OP1"], "O", "O"))

    # TRP2 = DTR without CP2/CO/OP1
    trp2 = _new_residue("TRP", 2, hetero=False)
    for at in dtr.get_atoms():
        if at.get_name().upper() in {"CP2", "CO", "OP1"}:
            continue
        trp2.add(
            _new_atom(
                at.get_name(),
                at.get_coord(),
                at.get_bfactor(),
                at.get_occupancy() or 1.0,
                at.element,
            )
        )

    new_chain = Chain.Chain(chain.id)
    new_chain.add(gly1)
    new_chain.add(trp2)

    # Middle residues: shift index by +1
    for new_idx, res in enumerate(residues[1:-1], start=3):
        het = str(res.id[0]).startswith("H_")
        new_res = _new_residue(
            res.get_resname(),
            new_idx,
            hetero=het and not Polypeptide.is_aa(res, standard=True),
        )
        for at in res.get_atoms():
            new_res.add(
                _new_atom(
                    at.get_name(),
                    at.get_coord(),
                    at.get_bfactor(),
                    at.get_occupancy() or 1.0,
                    at.element,
                )
            )
        new_chain.add(new_res)

    # Last: CYS -> ALA (drop SG)
    ala_idx = len(residues) + 1
    ala_last = _new_residue("ALA", ala_idx, hetero=False)
    for at in cys.get_atoms():
        if at.get_name().upper() == "SG":
            continue
        ala_last.add(
            _new_atom(
                at.get_name(),
                at.get_coord(),
                at.get_bfactor(),
                at.get_occupancy() or 1.0,
                at.element,
            )
        )
    new_chain.add(ala_last)

    return new_chain, True


def read_thioether(pdb_path: str) -> Tuple[Structure.Structure, List[str]]:
    """
    Convert all DTR…CYS thioether chains into pure native chains (GLY1, TRP2, …, ALA_last).
    Returns (converted_structure, converted_chain_ids).
    """
    parser = PDBParser(QUIET=True)
    src = parser.get_structure("thio", pdb_path)
    out = Structure.Structure("pure")
    converted: List[str] = []

    for model in src:
        new_model = Model.Model(model.id)
        for chain in model:
            new_chain, did = _build_chain_native_from_thioether(chain)
            new_model.add(new_chain)
            if did:
                converted.append(chain.id)
        out.add(new_model)
    return out, converted


from typing import Set

from Bio.PDB import Structure


def detect_thioether(structure: Structure.Structure) -> bool:
    """
    Quickly detect whether a Structure contains at least one 'thioether' chain,
    defined as a chain whose FIRST residue is HETATM DTR (with atoms CP2, CO, OP1)
    and whose LAST residue is standard CYS containing SG.

    Args:
        structure: Biopython Structure.

    Returns:
        bool: True if any chain matches the thioether signature; False otherwise.
    """
    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            if not residues:
                continue

            first = residues[0]
            last = residues[-1]

            # Check first residue: HETATM DTR
            # first.id is (hetfield, resseq, icode), e.g. ("H_DTR", 1, " ")
            hetfield_first = str(first.id[0])
            is_dtr = (
                first.get_resname().strip().upper() == "DTR"
            ) and hetfield_first.startswith("H_")
            if not is_dtr:
                continue

            # Verify DTR extra atoms exist
            first_atom_names: Set[str] = {
                a.get_name().strip().upper() for a in first.get_atoms()
            }
            if not {"CP2", "CO", "OP1"}.issubset(first_atom_names):
                continue

            # Check last residue: standard CYS with SG and non-hetero hetfield (" ")
            hetfield_last = last.id[0]
            is_cys = (last.get_resname().strip().upper() == "CYS") and (
                hetfield_last == " "
            )
            if not is_cys:
                continue

            last_atom_names: Set[str] = {
                a.get_name().strip().upper() for a in last.get_atoms()
            }
            if "SG" not in last_atom_names:
                continue

            # All criteria satisfied for this chain
            return True

    return False


def convert_thioether(structure: Structure.Structure) -> Structure.Structure:
    """
    Convert all DTR…CYS thioether chains into pure native chains (GLY1, TRP2, …, ALA_last).
    Returns converted_structure.
    """
    out = Structure.Structure("pure")

    for model in structure:
        new_model = Model.Model(model.id)
        for chain in model:
            new_chain, _ = _build_chain_native_from_thioether(chain)
            new_model.add(new_chain)
        out.add(new_model)
    return out


# ---------- inverse: native -> thioether ----------


def _build_chain_thioether_from_native(chain: Chain.Chain) -> Tuple[Chain.Chain, bool]:
    """
    Inverse operation for chains that look like: GLY1, TRP2, ..., ALA(last).
    Recreate DTR (HETATM) at N-terminus and CYS at C-terminus with SG from GLY.N.
    """
    residues = list(chain.get_residues())
    if len(residues) < 3:
        return chain.copy(), False

    first, second, last = residues[0], residues[1], residues[-1]
    if not (
        first.get_resname().upper() == "GLY"
        and second.get_resname().upper() == "TRP"
        and last.get_resname().upper() == "ALA"
    ):
        # Not a converted chain: keep as-is (renumbered)
        new_ch = Chain.Chain(chain.id)
        for i, res in enumerate(residues, start=1):
            het = str(res.id[0]).startswith("H_")
            new_res = _new_residue(res.get_resname(), i, hetero=het)
            for at in res.get_atoms():
                new_res.add(
                    _new_atom(
                        at.get_name(),
                        at.get_coord(),
                        at.get_bfactor(),
                        at.get_occupancy() or 1.0,
                        at.element,
                    )
                )
            new_ch.add(new_res)
        return new_ch, False

    for req in ("N", "CA", "C", "O"):
        if req not in {a.get_name().upper() for a in first.get_atoms()}:
            raise ValueError(
                f"Chain {chain.id}: GLY1 must contain {req} for inversion."
            )

    N, CA, C, O = first["N"], first["CA"], first["C"], first["O"]

    dtr = _new_residue("DTR", 1, hetero=True)
    # Copy all TRP2 atoms into DTR
    for at in second.get_atoms():
        dtr.add(
            _new_atom(
                at.get_name(),
                at.get_coord(),
                at.get_bfactor(),
                at.get_occupancy() or 1.0,
                at.element,
            )
        )
    # Add extra atoms from GLY
    dtr.add(_copy_atom_with_new_name(CA, "CP2", "C"))
    dtr.add(_copy_atom_with_new_name(C, "CO", "C"))
    dtr.add(_copy_atom_with_new_name(O, "OP1", "O"))

    new_chain = Chain.Chain(chain.id)
    new_chain.add(dtr)

    # Middle residues shift by -1
    for new_idx, res in enumerate(residues[2:-1], start=2):
        het = str(res.id[0]).startswith("H_")
        new_res = _new_residue(
            res.get_resname(),
            new_idx,
            hetero=het and not Polypeptide.is_aa(res, standard=True),
        )
        for at in res.get_atoms():
            new_res.add(
                _new_atom(
                    at.get_name(),
                    at.get_coord(),
                    at.get_bfactor(),
                    at.get_occupancy() or 1.0,
                    at.element,
                )
            )
        new_chain.add(new_res)

    # ALA -> CYS, add SG at GLY.N coordinate
    cys_idx = len(residues) - 1
    cys_last = _new_residue("CYS", cys_idx, hetero=False)
    for at in last.get_atoms():
        cys_last.add(
            _new_atom(
                at.get_name(),
                at.get_coord(),
                at.get_bfactor(),
                at.get_occupancy() or 1.0,
                at.element,
            )
        )
    cys_last.add(_copy_atom_with_new_name(N, "SG", "S"))
    new_chain.add(cys_last)

    return new_chain, True


def write_thioether(
    structure: Structure.Structure,
    out_path: str,
    chain_ids: str | List[str] | None = None,
):
    """
    Invert selected chains of a pure-protein Structure back to the original thioether form and write PDB.
    """
    if chain_ids is None:
        chain_ids = ["A"]
    elif isinstance(chain_ids, str):
        chain_ids = [chain_ids]

    new_struct = Structure.Structure("thio_reverted")
    for model in structure:
        new_model = Model.Model(model.id)
        for chain in model:
            if chain.id in set(chain_ids):
                reverted_chain, _ = _build_chain_thioether_from_native(chain)
            else:
                # pass-through (renumbered) copy
                residues = list(chain.get_residues())
                reverted_chain = Chain.Chain(chain.id)
                for i, res in enumerate(residues, start=1):
                    het = str(res.id[0]).startswith("H_")
                    new_res = _new_residue(res.get_resname(), i, hetero=het)
                    for at in res.get_atoms():
                        new_res.add(
                            _new_atom(
                                at.get_name(),
                                at.get_coord(),
                                at.get_bfactor(),
                                at.get_occupancy() or 1.0,
                                at.element,
                            )
                        )
                    reverted_chain.add(new_res)
            new_model.add(reverted_chain)
        new_struct.add(new_model)
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_path)
