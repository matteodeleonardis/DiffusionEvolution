import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution

output_root = ARGS[1]
file_model_scores = ARGS[2:end]

reprint_inference(
    output_root=output_root, 
    file_model_scores=file_model_scores
)