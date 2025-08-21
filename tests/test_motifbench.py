"""
Tests for motif contig placement.

Author: Tianyu Lu
"""
import numpy as np

from protpardelle.data.motif import contig_to_motif_placement


def test_contig_to_motif_placement():
    all_motif_indices, all_placement_specs, all_full_specs, all_total_lengths = contig_to_motif_placement("5-20;A16-35;10-25;A52-71;5-20", [60, 105], 100)

    assert len(all_motif_indices) == 100
    assert len(all_placement_specs) == 100
    assert len(all_full_specs) == 100
    assert len(all_total_lengths) == 100

    for i, motif_placement in enumerate(all_full_specs):
        curr_motif_idx = all_motif_indices[i]
        curr_placement = all_placement_specs[i]
        curr_total_length = all_total_lengths[i]

        motif_seglens, reconstructed_placement, scaffold_seglens, all_seglens = [], [], [], []
        for segment in motif_placement.split('/'):
            if segment[0].isalpha():
                reconstructed_placement.append(segment[0])
                start, end = segment[1:].split('-')
                curr_len = int(end) - int(start) + 1
                motif_seglens.append(curr_len)
            else:
                curr_len = int(segment)
                scaffold_seglens.append(curr_len)
                reconstructed_placement.append(str(curr_len))
            all_seglens.append(curr_len)

        # check that returned info is self-consistent
        assert sum(all_seglens) == curr_total_length
        
        reconstructed_placement = '/'.join(reconstructed_placement)
        assert reconstructed_placement == curr_placement

        full_idx = np.arange(curr_total_length)
        scaffold_idx = []
        start_idx = 0
        for i, seglen in enumerate(all_seglens):
            if i % 2 == 0:
                scaffold_idx.extend(range(start_idx, start_idx + seglen))
            start_idx += seglen
        reconstructed_motif_idx = np.delete(full_idx, scaffold_idx)
        assert np.all(reconstructed_motif_idx == curr_motif_idx)

        # check that returned info satisfy motif contig constraints
        constraints = [range(5, 20 + 1), "A16-35", range(10, 25 + 1), "A52-71", range(5, 20 + 1)]
        for i, segment in enumerate(motif_placement.split('/')):
            if segment[0].isalpha():
                assert segment in constraints[i]
            else:
                assert int(segment) in constraints[i]
        
        assert curr_total_length in range(60, 105 + 1)
