#!/bin/bash

dir_files=$1
gamma=$2

input_fasta="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1_clean.fasta"
plmdca_dir="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/plmdca"
output_root="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/${dir_files}/plmdca"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ] || [ ! -e "${output_root}.score.jld2" ]; then
    ln -sf "${plmdca_dir}/plmdca.score.jld2" "${output_root}.score.jld2"
fi

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "$(dirname "$0")/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)


julia $(dirname "$0")/plmdca_analysis.jl $input_fasta $plmdca_dir $output_root $gamma "${model_score_files[@]}"