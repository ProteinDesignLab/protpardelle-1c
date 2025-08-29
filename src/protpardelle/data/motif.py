"""Functions for motif contig parsing.

Author: Tianyu Lu
"""

import random
import string

import numpy as np


def motif_placement_to_motif_idx(
    motif_placement: str, motif_id_to_length: dict[str, int]
) -> list[int]:
    """
    Convert a motif placement string to 0-indexed motif index
    Also need a mapping from motif segment id to motif segment length

    E.g. "12/B/40/A/41" -> [12, 13, 14, 55, 56, 57, 58]
    """
    motif_idx = []

    curr_idx = 0

    for s in motif_placement.split("/"):
        if s.isalpha():  # motif segment
            motif_length = motif_id_to_length[s]
            m_idx = list(range(curr_idx, curr_idx + motif_length))
            motif_idx.extend(m_idx)
            curr_idx += motif_length
        else:  # scaffold segment
            curr_idx += int(s)

    return motif_idx


def motif_idx_to_motif_placement(motif_idx, total_length: int):
    """
    Assume each motif index is one chain for linear_sum_assignment based dynamic indexing
    """
    idx_sorted = np.argsort(motif_idx)
    chains = string.ascii_uppercase[: len(motif_idx)]
    chains = [chains[i] for i in idx_sorted]
    motif_idx_sorted = np.sort(motif_idx)
    motif_placement = [str(motif_idx_sorted[0]), chains[0]]
    for i, idx in enumerate(motif_idx_sorted[1:], start=1):
        motif_placement.extend((str(idx - motif_idx_sorted[i - 1] - 1), chains[i]))
    motif_placement.append(str(total_length - motif_idx_sorted[-1] - 1))
    scaffold_length = sum(int(sl) for sl in motif_placement if not str(sl).isalpha())
    assert scaffold_length + len(motif_idx) == total_length
    return "/".join(motif_placement)


def remaining_min_possible(
    num_remaining, parsed_segments, scaffold_indices, current_si
):
    return sum(
        parsed_segments[scaffold_indices[current_si + j + 1]][1][0]
        for j in range(num_remaining)
    )


def remaining_max_possible(
    num_remaining, parsed_segments, scaffold_indices, current_si
):
    return sum(
        parsed_segments[scaffold_indices[current_si + j + 1]][1][1]
        for j in range(num_remaining)
    )


def contig_to_motif_placement(contig: str, length_range: list[int], num_samples: int):
    """
    Example:
    - a contig formatting like this: 10-25;B1-7;20-60;A1-7;20-60
    - length_range: [100, 150]
    - num_samples: 2

    Sample from each scaffold length such that the sum, plus motif lengths equals total_length

    Return the 0-indexed motif index of all samples,
    e.g. [[12, 13, 14, 55, 56, 57, 58], [22, 23, 24, 50, 51, 52, 53]]
    also the motif placement specification,
    e.g. the above would look like ["12/B/40/A/41", "22/B/25/A/46"]
    and the full motif placement with residue indices for each motif segment,
    e.g. ["12/B1-7/40/A1-7/41", "22/B1-7/25/A1-7/46"]
    and a list of the total length for each sample
    """

    def is_motif(segment: str) -> bool:
        return any(c.isalpha() for c in segment)

    parts = contig.split(";")
    segment_types = ["motif" if is_motif(p) else "scaffold" for p in parts]

    # Parse segments
    parsed_segments = []
    motif_total_len = 0
    for typ, p in zip(segment_types, parts):
        if typ == "scaffold":
            start, end = map(int, p.split("-"))
            parsed_segments.append(("scaffold", (start, end)))
        else:
            chain = p[0]
            start, end = map(int, p[1:].split("-"))
            m_len = end - start + 1
            motif_total_len += m_len
            parsed_segments.append(("motif", (chain, start, end)))

    min_total, max_total = length_range
    min_scaffold_sum = min_total - motif_total_len
    max_scaffold_sum = max_total - motif_total_len

    scaffold_indices = [
        i for i, (typ, _) in enumerate(parsed_segments) if typ == "scaffold"
    ]

    all_motif_indices = []
    all_placement_specs = []
    all_full_specs = []
    all_total_lengths = []

    for _ in range(num_samples):
        # Sample scaffold lengths jointly so sum is within range
        scaffold_lengths = {}
        remaining_min = min_scaffold_sum
        remaining_max = max_scaffold_sum
        remaining_scaffolds = len(scaffold_indices)

        for si, seg_idx in enumerate(scaffold_indices):
            min_len, max_len = parsed_segments[seg_idx][1]

            if si < len(scaffold_indices) - 1:
                # Adjust bounds so remaining segments can still fit
                min_possible_here = max(
                    min_len,
                    remaining_min
                    - (
                        remaining_max_possible(
                            remaining_scaffolds - 1,
                            parsed_segments,
                            scaffold_indices,
                            si,
                        )
                    ),
                )
                max_possible_here = min(
                    max_len,
                    remaining_max
                    - (
                        remaining_min_possible(
                            remaining_scaffolds - 1,
                            parsed_segments,
                            scaffold_indices,
                            si,
                        )
                    ),
                )
                length = random.randint(min_possible_here, max_possible_here)
            else:
                # Last scaffold must make total valid
                min_possible_here = max(min_len, remaining_min)
                max_possible_here = min(max_len, remaining_max)
                length = random.randint(min_possible_here, max_possible_here)

            scaffold_lengths[seg_idx] = length
            remaining_min -= length
            remaining_max -= length
            remaining_scaffolds -= 1

        # Build placement
        pos = 0
        motif_indices = []
        placement_spec = []
        full_spec = []
        total_len = 0
        prev_scaffold_len = 0

        for seg_idx, (typ, val) in enumerate(parsed_segments):
            if typ == "scaffold":
                seg_len = scaffold_lengths[seg_idx]
                prev_scaffold_len = seg_len
                pos += seg_len
                total_len += seg_len
                if seg_idx == len(parsed_segments) - 1:
                    placement_spec.append(str(seg_len))
                    full_spec.append(str(seg_len))
            else:  # motif
                chain, m_start, m_end = val
                m_len = m_end - m_start + 1
                indices = list(range(pos, pos + m_len))
                motif_indices.extend(indices)
                if pos != 0:
                    placement_spec.append(str(prev_scaffold_len))
                    full_spec.append(f"{prev_scaffold_len}/{chain}{m_start}-{m_end}")
                else:
                    full_spec.append(f"{chain}{m_start}-{m_end}")

                placement_spec.append(chain)
                pos += m_len
                total_len += m_len

        if not placement_spec:
            placement_spec.append(str(total_len))
        if not full_spec:
            full_spec.append(str(total_len))

        all_motif_indices.append(motif_indices)
        all_placement_specs.append("/".join(placement_spec))
        all_full_specs.append("/".join(full_spec))
        all_total_lengths.append(total_len)

    return all_motif_indices, all_placement_specs, all_full_specs, all_total_lengths
