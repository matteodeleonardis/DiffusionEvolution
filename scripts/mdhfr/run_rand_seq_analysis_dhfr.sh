#!/bin/bash

dir_files=$1
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#output_root="${this_dir}/../../results/mdhfr/${dir_files}/mdhfr_analysis_d_"

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "mdhfr_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_pars_files < <(
  find "${this_dir}/../../results/mdhfr/$dir_files" -type f -name 'mdhfr_analysis_d_*.pars.jld2' \
  | awk 'match($0, /\/mdhfr_analysis_d_([0-9]+)\/mdhfr_analysis_d_[0-9]+\.pars\.jld2$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)

#debug_file=("/home/students/s301803/diffusion_evolution/dev/analysis/mdhfr/run0/mdhfr_analysis_d_50/mdhfr_analysis_d_50.scores.zerosumgauge_apc.tsv")

julia "${this_dir}/rand_seq_analysis_dhfr.jl" "${model_pars_files[@]}"

#"${debug_file[@]}"