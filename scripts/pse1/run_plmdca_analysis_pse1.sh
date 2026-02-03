#!/bin/bash

dir_files=$1

input_fasta="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1_clean.fasta"
output_root="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/${dir_files}/plmdca"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ]; then
    ln -s "/home/students/s301803/CODE/DiffusionEvolution/results/pse1/plmdca/plmdca.score.jld2" "${output_root}.score.jld2"
fi

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "$PWD/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)


julia plmdca_analysis.jl $input_fasta $output_root "${model_score_files[@]}"