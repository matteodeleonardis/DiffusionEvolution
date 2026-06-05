#!/bin/bash

dir_files=$1
this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t model_pars_files < <(
  find "${this_dir}/../../results/pse1/$dir_files" -type f -name 'pse1_analysis_d_*.pars.jld2' \
  | awk 'match($0, /\/pse1_analysis_d_([0-9]+)\/pse1_analysis_d_[0-9]+\.pars\.jld2$/, m) {print m[1] "\t" $0}' \
  | sort -n -k1,1 \
  | cut -f2-
)

julia "${this_dir}/rand_seq_analysis_pse1.jl" "${model_pars_files[@]}"
