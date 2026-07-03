#!/bin/bash
# Script to concatenate all chroma design output CSV files after array jobs complete
# Usage: ./concat_chroma_outputs.sh [OUTPUT_DIR]
# If OUTPUT_DIR is not provided, uses the default directory

# Default output directory
DEFAULT_OUTPUT_DIR="/scratch/users/tianyulu/allatom-design-sampling-results/8fns_conformations/chroma_design_output"

# Use provided argument or default
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"
FINAL_OUTPUT="${OUTPUT_DIR}/all_designs_combined.csv"

echo "Using output directory: ${OUTPUT_DIR}"

# Check if output directory exists
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: Output directory does not exist: ${OUTPUT_DIR}"
    exit 1
fi

# Find all design output CSV files
CSV_FILES=($(ls ${OUTPUT_DIR}/input_chunk_*_design_output.csv 2>/dev/null | sort -V))

if [ ${#CSV_FILES[@]} -eq 0 ]; then
    echo "Error: No output CSV files found in ${OUTPUT_DIR}"
    exit 1
fi

echo "Found ${#CSV_FILES[@]} output CSV files"

# Get header from first file
head -1 "${CSV_FILES[0]}" > "${FINAL_OUTPUT}"

# Concatenate all files (without headers)
for csv_file in "${CSV_FILES[@]}"; do
    echo "Adding: ${csv_file}"
    tail -n +2 "${csv_file}" >> "${FINAL_OUTPUT}"
done

# Count total rows
TOTAL_ROWS=$(wc -l < "${FINAL_OUTPUT}")
echo ""
echo "Concatenation complete!"
echo "Output file: ${FINAL_OUTPUT}"
echo "Total rows (including header): ${TOTAL_ROWS}"
echo "Total designs: $((TOTAL_ROWS - 1))"
