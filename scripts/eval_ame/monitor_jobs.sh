#!/bin/bash
# Quick health-check on AME sampling array jobs.
# Usage: bash scripts/eval_ame/monitor_jobs.sh [jobid1 jobid2 ...]

JOBS="${@:-22810088 22810090 22810091 22810092}"
LOGS=/scratch/users/tianyulu/slurm_log_files

echo "=== Job state ==="
squeue -u tianyulu --format="%.10i %.18j %.10T %.10M %.12l %.10P %R" | grep -E "ame_|JOBID" | head -20

echo ""
echo "=== Per-task tail (out + err) ==="
for j in $JOBS; do
    for arr in 0 1 2 3 4 5 6 7; do
        out="${LOGS}/ame_samp-${j}_${arr}.out"
        err="${LOGS}/ame_samp-${j}_${arr}.err"
        [[ -f "$out" || -f "$err" ]] || continue
        echo ""
        echo "--- job ${j}_${arr} ---"
        if [[ -f "$out" ]]; then
            tail -3 "$out" | grep -v "^$" | head -3
        fi
        # Check for tracebacks / errors
        if [[ -f "$err" ]] && grep -qE "Traceback|Error|MemoryError|CUDA error" "$err"; then
            echo "  !! ERROR DETECTED:"
            grep -E "Traceback|Error|MemoryError|CUDA error" "$err" | head -3
        fi
    done
done

echo ""
echo "=== Output PDB count by yaml ==="
SAMP=/scratch/users/tianyulu/protpardelle-1c-sampling-results
for yaml in ame_cc91_tip_condmask ame_cc91_tip_condmask_large ame_cc138 ame_cc137; do
    n=$(find ${SAMP}/${yaml} -name "sample_*.pdb" 2>/dev/null | wc -l)
    echo "  ${yaml}: ${n} sample PDBs written"
done
