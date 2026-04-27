#!/bin/bash

dir_files=$1
this_dir="$(dirname ${BASH_SOURCE[0]})"
in_nat="${this_dir}/../../data/dhfr/mDHFR_clean.fasta"
in_wt="${this_dir}/../../data/dhfr/mDHFR.fasta"
in_r1="${this_dir}/../../data/dhfr/Round1_Q15_C10_aa.aln"
in_r2="${this_dir}/../../data/dhfr/Round2_Q15_C10_aa.aln"
in_r3="${this_dir}/../../data/dhfr/Round3_Q15_C10_aa.aln"
in_r4="${this_dir}/../../data/dhfr/Round4_Q15_C10_aa.aln"
in_r5="${this_dir}/../../data/dhfr/Round5_Q15_C10_aa.aln"
in_r15="${this_dir}/../../data/dhfr/Gen15_aa.aln"
output_root="${this_dir}/../../results/mdhfr/${dir_files}/ev_couplings"
output_dir=$(dirname "$output_root")
mkdir -p $output_dir
if [ ! -L "${output_root}.score.jld2" ] || [ ! -e "${output_root}.score.jld2" ]; then
    ln -sf "${this_dir}/../../results/mdhfr/ev_couplings/ev_couplings.score.jld2" "${output_root}.score.jld2"
fi
evc_score_dir="${this_dir}/../../results/mdhfr/ev_couplings/"
mkdir -p $evc_score_dir

#mapfile -t model_score_files < <(find "$PWD/run0" -type f -name "mdhfr_analysis_d_*.scores.zerosumgauge_apc.tsv")
mapfile -t model_score_files < <(
  find "${this_dir}/../../results/mdhfr/$dir_files" -type f -name 'mdhfr_analysis_d_*.scores.zerosumgauge_apc.tsv' \
  | awk 'match($0, /\/mdhfr_analysis_d_([0-9]+)\/mdhfr_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)

#debug_file=("/home/students/s301803/diffusion_evolution/dev/analysis/mdhfr/run0/mdhfr_analysis_d_50/mdhfr_analysis_d_50.scores.zerosumgauge_apc.tsv")

julia "${this_dir}/ev_couplings_analysis.jl" $in_nat \
    $in_r1 $in_wt $in_r2 $in_r3 $in_r4 $in_r5 $in_r15 \
    $evc_score_dir $output_root "${model_score_files[@]}"

#"${debug_file[@]}"