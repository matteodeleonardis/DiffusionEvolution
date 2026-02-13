import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution

contacts_file = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/contact_map.jld2"
input_fasta = ARGS[1]
wt_fasta = ARGS[2]
output_root = ARGS[3]
low_rank_mf_dir = ARGS[4]
file_model_scores = ARGS[5:end]

run_low_rank_mf_analysis_pse1(
    input_fasta=input_fasta, 
    wt_fasta=wt_fasta, 
    contacts_file=contacts_file, 
    output_root=output_root, 
    low_rank_mf_dir=low_rank_mf_dir, 
    file_model_scores=file_model_scores
)