#!/bin/bash

dir_files=$1
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${this_dir}/../../results/pse1/${dir_files}/pse1_analysis_d_"

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "${this_dir}/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)

#debug_file=("/home/students/s301803/diffusion_evolution/dev/analysis/pse1/run0/pse1_analysis_d_50/pse1_analysis_d_50.scores.zerosumgauge_apc.tsv")

julia "${this_dir}/reprint_inference.jl" $output_root "${model_score_files[@]}"

#"${debug_file[@]}"