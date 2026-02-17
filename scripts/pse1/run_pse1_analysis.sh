#!/bin/bash

code="$1"
env_path="/home/students/s301803/CODE/DiffusionEvolution/scripts"
max_threads=12
shift
d_val=("$@")

for d in "${d_val[@]}"
do
    session="${code}-pse1_d_${d}"
    outdir="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/${code}/pse1_analysis_d_${d}"
    output_root="${outdir}/pse1_analysis_d_${d}"

    stdout_log="${outdir}/stdout.log"
    stderr_log="${outdir}/stderr.log"

    mkdir -p "$outdir"

    tmux new-session -d -s "$session" \
        "OMP_NUM_THREADS=${max_threads} \
        OPENBLAS_NUM_THREADS=${max_threads} \
        MKL_NUM_THREADS=${max_threads} \
        VECLIB_MAXIMUM_THREADS=${max_threads} \
        JULIA_NUM_THREADS=1 \
        julia --project=${env_path} pse1_analysis.jl ${d} ${output_root} \
         > ${stdout_log} 2> ${stderr_log}"
done
