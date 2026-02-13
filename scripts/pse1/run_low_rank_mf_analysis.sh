#!/bin/bash

dir_files=$1
input_fasta="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1_clean.fasta"
input_wt="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1.fas"
output_root="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/${dir_files}/low_rank_mf"
low_rank_dir="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/low_rank_mf/"
mkdir -p $low_rank_dir

mapfile -t model_score_files < <(
  find "$(dirname "$0")/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)


julia $(dirname "$0")/low_rank_pse1_analysis.jl $input_fasta $input_wt $output_root $low_rank_dir "${model_score_files[@]}"