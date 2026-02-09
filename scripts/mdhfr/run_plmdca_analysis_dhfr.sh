#!/bin/bash

dir_files=$1
input_fasta="/home/students/s301803/CODE/DiffusionEvolution/data/dhfr/mDHFR_clean.fasta"
output_root="/home/students/s301803/CODE/DiffusionEvolution/results/mdhfr/${dir_files}/plmdca"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ] || [ ! -e "${output_root}.score.jld2" ]; then
    ln -sf "/home/students/s301803/CODE/DiffusionEvolution/results/mdhfr/plmdca/plmdca.score.jld2" "${output_root}.score.jld2"
fi

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "mdhfr_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "$(dirname "$0")/../../results/mdhfr/$dir_files" -type f -name 'mdhfr_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/mdhfr_analysis_d_([0-9]+)\/mdhfr_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)

#debug_file=("/home/students/s301803/diffusion_evolution/dev/analysis/mdhfr/run0/mdhfr_analysis_d_50/mdhfr_analysis_d_50.scores.zerosumgauge_apc.tsv")

julia $(dirname "$0")/plmdca_analysis.jl $input_fasta $output_root "${model_score_files[@]}"

#"${debug_file[@]}"