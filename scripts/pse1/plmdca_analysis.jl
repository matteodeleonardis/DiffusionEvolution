import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution


input_fasta = ARGS[1]
plmdca_dir = ARGS[2]
output_root = ARGS[3]
fixed  = ARGS[4] == "no"
model_scores = ARGS[5:end]
contacts_file = joinpath(@__DIR__, "../../data/pse1/contact_map.jld2")

run_plmdca_analysis_pse1(
    input_fasta=input_fasta,
    plmdca_dir=plmdca_dir,
    contacts_file=contacts_file,
    output_root=output_root,
    fixed = fixed,
    file_model_scores=model_scores
)

