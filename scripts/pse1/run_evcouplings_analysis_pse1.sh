#!/bin/bash

dir_files=$1

this_dir="$(dirname ${BASH_SOURCE[0]})"
in_nat="${this_dir}/../../data/pse1/PSE1_clean.fasta"
in_wt="${this_dir}/../../data/pse1/PSE1.fasta"
in_r10="${this_dir}/../../data/pse1/Rnd10.fasta"
in_r20="${this_dir}/../../data/pse1/Rnd20_init.fasta"
output_root="${this_dir}/../../results/pse1/${dir_files}/ev_couplings"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ] || [ ! -e "${output_root}.score.jld2" ]; then
    ln -sf "${this_dir}/../../results/pse1/ev_couplings/ev_couplings.score.jld2" "${output_root}.score.jld2"
fi

evc_score_dir="${this_dir}/../../results/pse1/ev_couplings/"
mkdir -p $evc_score_dir

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "${this_dir}/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)


julia "${this_dir}/ev_couplings_analysis.jl" $in_nat \
    $in_wt $in_r10 $in_r20 \
    $evc_score_dir $output_root "${model_score_files[@]}"