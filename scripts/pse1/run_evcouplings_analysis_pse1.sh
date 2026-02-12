#!/bin/bash

dir_files=$1

in_nat="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1_clean.fasta"
in_wt="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1.fas"
in_r10="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/Rnd10.fas"
in_r20="/home/students/s301803/CODE/DiffusionEvolution/data/pse1/Rnd20_init.fas"
output_root="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/${dir_files}/ev_couplings"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ] || [ ! -e "${output_root}.score.jld2" ]; then
    ln -sf "/home/students/s301803/CODE/DiffusionEvolution/results/pse1/ev_couplings/ev_couplings.score.jld2" "${output_root}.score.jld2"
fi

evc_score_dir="/home/students/s301803/CODE/DiffusionEvolution/results/pse1/ev_couplings/"
mkdir -p $evc_score_dir

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "$(dirname "$0")/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)


julia $(dirname "$0")/ev_couplings_analysis.jl $in_nat \
    $in_wt $in_r10 $in_r20 \
    $evc_score_dir $output_root "${model_score_files[@]}"