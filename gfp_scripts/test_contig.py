def fixpos_to_contig(pdb_length, fixpos_ints):
    """
    Convert fixed positions into a motif contig string.
    
    Args:
        pdb_length: Total length of the PDB structure
        fixpos_ints: List of fixed position integers (1-indexed)
    
    Returns:
        Contig string in the format: length;Apos-pos;length;...
    """
    if not fixpos_ints:
        return f"0-{pdb_length - 1}"
    
    # Sort fixed positions
    fixpos_ints = sorted(fixpos_ints)
    
    contig_parts = []
    current_pos = 0
    
    i = 0
    while i < len(fixpos_ints):
        # Add gap before fixed position(s)
        gap_length = fixpos_ints[i] - 1 - current_pos
        if gap_length > 0:
            contig_parts.append(f"{gap_length}-{gap_length}")
        
        # Find consecutive fixed positions
        start_fix = fixpos_ints[i]
        end_fix = fixpos_ints[i]
        
        while i + 1 < len(fixpos_ints) and fixpos_ints[i + 1] == fixpos_ints[i] + 1:
            i += 1
            end_fix = fixpos_ints[i]
        
        # Add fixed position(s)
        contig_parts.append(f"A{start_fix}-{end_fix}")
        current_pos = end_fix
        i += 1
    
    # Add remaining positions after last fixed position
    remaining = pdb_length - current_pos
    if remaining > 0:
        contig_parts.append(f"{remaining}-{remaining}")
    
    return ";".join(contig_parts)


# Example usage
pdb_length = 230
fixpos_ints = [45, 125, 126]

contig = fixpos_to_contig(pdb_length, fixpos_ints)
print(f"Input: pdb_length={pdb_length}, fixpos_ints={fixpos_ints}")
print(f"Output: {contig}")

# Additional test cases
print("\nAdditional examples:")
print(fixpos_to_contig(100, [10, 20, 30]))
print(fixpos_to_contig(50, [1, 2, 3, 48, 49, 50]))
print(fixpos_to_contig(200, [100]))