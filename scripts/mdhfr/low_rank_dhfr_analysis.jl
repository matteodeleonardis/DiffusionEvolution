import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

contacts_file = joinpath(@__DIR__, "..", "..", "data/dhfr/contact_map.npy")
input_fasta = ARGS[1]
wt_fasta = ARGS[2]
output_root = ARGS[3]
low_rank_mf_dir = ARGS[4]
file_model_scores = ARGS[5:end]

run_low_rank_mf_analysis_dhfr(
    input_fasta=input_fasta, 
    wt_fasta=wt_fasta, 
    contacts_file=contacts_file, 
    output_root=output_root, 
    low_rank_mf_dir=low_rank_mf_dir, 
    file_model_scores=file_model_scores
)