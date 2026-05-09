import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution

input_fasta = ARGS[1]
wt_fasta = ARGS[2]
output_root = ARGS[3]
file_model_scores = ARGS[4:end]

J_divergence(
    input_fasta=input_fasta, 
    wt_fasta=wt_fasta, 
    output_root=output_root, 
    file_model_scores=file_model_scores
)