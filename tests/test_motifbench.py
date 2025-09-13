"""A pytest module for motif contig placement utilities.

This module tests `contig_to_motif_placement` which expands a motif/scaffold
specification string like:

    5-20;A16-35;10-25;A52-71;5-20

into concrete motif residue index arrays, placement specifications, and total
lengths. The function returns lists of length `n_samples` containing sampled
instantiations satisfying the variable scaffold length ranges.

Test strategy:
1. Happy path with mixed scaffold ranges and motif segments.
2. Edge case: single motif segment with zero-length flanking scaffold.
3. Determinism of output lengths bounds.

The helper `_parse_full_spec` reconstructs lengths to validate returned indices.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from protpardelle.data.motif import contig_to_motif_placement


def _parse_full_spec(spec: str) -> tuple[list[int], str, list[int]]:
    """Parse a full placement spec (alternating scaffold length / motif segment).

    Returns tuple(all_seglens, reconstructed_simple_spec, motif_segment_lengths)
    where reconstructed_simple_spec uses only the chain letter for motif segments
    and raw integers for scaffold segments (mirrors `all_placement_specs`).

    Args:
        spec (str): The full placement specification string.

    Returns:
        tuple[list[int], str, list[int]]: A tuple containing:
            - all_seglens: A list of all segment lengths.
            - reconstructed_simple_spec: The reconstructed simple specification string.
            - motif_segment_lengths: A list of motif segment lengths.
    """

    all_seglens = []
    simple_parts = []
    motif_lens = []
    for segment in spec.split("/"):
        if segment[0].isalpha():
            simple_parts.append(segment[0])
            start, end = segment[1:].split("-")
            seg_len = int(end) - int(start) + 1
            motif_lens.append(seg_len)
        else:
            seg_len = int(segment)
            simple_parts.append(str(seg_len))
        all_seglens.append(seg_len)
    return all_seglens, "/".join(simple_parts), motif_lens


def _reconstruct_motif_indices(total_len: int, seglens: Sequence[int]) -> np.ndarray:
    """Given alternating scaffold/motif segment lengths, recover motif index array."""
    full_idx = np.arange(total_len)
    scaffold_idx: list[int] = []
    start = 0
    for pos, seg_len in enumerate(seglens):
        if pos % 2 == 0:  # scaffold segment positions
            scaffold_idx.extend(range(start, start + seg_len))
        start += seg_len
    return np.delete(full_idx, scaffold_idx)


@pytest.mark.parametrize(
    "spec,length_range,n_samples,constraints",
    [
        (
            "5-20;A16-35;10-25;A52-71;5-20",
            [60, 105],
            32,  # fewer samples for faster test
            [
                range(5, 21),
                "A16-35",
                range(10, 26),
                "A52-71",
                range(5, 21),
            ],
        ),
    ],
)
def test_contig_to_motif_placement_happy(
    spec: str,
    length_range: list[int],
    n_samples: int,
    constraints: list[range | str],
) -> None:
    """Test the contig_to_motif_placement function with a happy path scenario.

    Args:
        spec (str): The full placement specification string.
        length_range (list[int]): A list containing the minimum and maximum length of the contig.
        n_samples (int): The number of samples to generate.
        constraints (list[range  |  str]): A list of constraints for each segment in the spec.
    """
    mot_idx_list, placement_specs, full_specs, total_lengths = (
        contig_to_motif_placement(spec, length_range, n_samples)
    )

    assert len(mot_idx_list) == n_samples
    assert len(placement_specs) == n_samples
    assert len(full_specs) == n_samples
    assert len(total_lengths) == n_samples

    for mot_idx, placement_simple, full_spec, total_len in zip(
        mot_idx_list, placement_specs, full_specs, total_lengths
    ):
        seglens, reconstructed_simple, _ = _parse_full_spec(full_spec)
        assert reconstructed_simple == placement_simple
        assert sum(seglens) == total_len
        # motif indices reconstruct
        recon = _reconstruct_motif_indices(total_len, seglens)
        assert np.array_equal(recon, mot_idx)
        # length bounds respected
        assert length_range[0] <= total_len <= length_range[1]
        # constraint satisfaction
        for cpos, segment in enumerate(full_spec.split("/")):
            constraint = constraints[cpos]
            if segment[0].isalpha():
                # motif segment must match exact motif constraint string
                assert isinstance(constraint, str)
                assert segment == constraint
            else:
                # scaffold segment length must lie inside allowed range
                assert isinstance(constraint, range)
                assert int(segment) in constraint


def test_single_motif_no_scaffold() -> None:
    """Test the contig_to_motif_placement function with a single motif and no scaffold."""
    mot_idx_list, placement_specs, full_specs, total_lengths = (
        contig_to_motif_placement("A1-5", [5, 5], 3)
    )
    assert all(tl == 5 for tl in total_lengths)
    for mot_idx, placement_simple, full_spec in zip(
        mot_idx_list, placement_specs, full_specs
    ):
        assert placement_simple == "A"
        _, reconstructed_simple, motif_lens = _parse_full_spec(full_spec)
        assert reconstructed_simple == "A"
        assert motif_lens == [5]
        # motif occupies entire range
        assert np.array_equal(mot_idx, np.arange(5))


def test_length_bounds_respected() -> None:
    """Test that the length bounds are respected in the placement.

    Randomized spec with scaffold flexibility. Original (25,40) range was infeasible:
    motif length = (10..19) -> 10 residues. Scaffold min sum = 3 + 1 = 4 giving min total 14.
    To test bounds we pick a feasible window that contains achievable totals.
    """

    mot_idx_list, _, _, total_lengths = contig_to_motif_placement(
        "3-6;A10-19;1-4", [14, 25], 16
    )
    for tl in total_lengths:
        assert 14 <= tl <= 25
    # basic sanity: motif indices strictly increasing
    for mot_idx in mot_idx_list:
        assert np.all(np.diff(mot_idx) > 0)
