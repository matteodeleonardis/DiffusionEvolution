import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution

input_fasta = ARGS[1]
output_root = ARGS[2]
fixed  = ARGS[3] == "no"
model_scores = ARGS[4:end]
contacts_file = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/contact_map.jld2"

run_plmdca_analysis_pse1(
    input_fasta=input_fasta,
    contacts_file=contacts_file,
    output_root=output_root,
    fixed = fixed,
    file_model_scores=model_scores
)

