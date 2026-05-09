# DiffusionEvolution

To perform the whole pipeline of analysis execute the following commands.

## Preprocess data
To preprocess natural sequences files:

`data/msa_preprocess/preprocess_dhfr.sh`

`data/msa_preprocess/preprocess_pse1.sh`

## Train OU model
Train OU model on experimental data:

`bash scripts/mdhfr/dhfr_analysis.jl {output_directory_name} {d1 d2 d3 ...}`

`bash scripts/pse1/pse1_analysis.jl {output_directory_name} {d1 d2 d3 ...}`

name of the output (just the name) `{output_directory_name}`: `test`/`run_0`...

list of the values for the dimension of the latent space `{d1 d2 d3 ...}`: `1 2 3 4`

## PlmDCA analysis

`bash scripts/mdhfr/run_plmdca_analysis_dhfr.sh {output_directory_name} no`

`bash scripts/pse1/run_plmdca_analysis_pse1.sh {output_directory_name} no`

## Low-rank analysis 

`bash scripts/mdhfr/run_low_rank_mf_analysis.sh {output_directory_name}`

`bash scripts/pse1/run_low_rank_mf_analysis.sh {output_directory_name}`

## Combined-MSA analysis (EVCouplings)

`bash scripts/mdhfr/run_evcouplings_analysis_dhfr.sh {output_directory_name}`

`bash scripts/pse1/run_evcouplings_analysis_pse1.sh {output_directory_name}`

## Tuning latent space dimension d

`bash scripts/mdhfr/run_J_divergence.sh {output_directory_name}`

`bash scripts/pse1/run_J_divergence.sh {output_directory_name}`