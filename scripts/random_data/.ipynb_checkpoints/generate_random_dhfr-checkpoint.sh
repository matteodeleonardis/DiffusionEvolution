#!/bin/bash

output_root=/home/students/s301803/CODE/DiffusionEvolution/data/random_dhfr/random_dhfr
mkdir -p "$(dirname $output_root)"

fasta_wt="/home/students/s301803/CODE/DiffusionEvolution/data/dhfr/mDHFR.fasta"
pmut=0.01
n_rounds=15
n_seqs=1000

julia generate_random.jl "$fasta_wt" "$pmut" "$n_rounds" "$n_seqs" "$output_root"