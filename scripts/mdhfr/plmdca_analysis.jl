import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution

input_fasta = ARGS[1]
output_root = ARGS[2]
model_scores = ARGS[3:end]
contacts_file = "/home/students/s301803/dhfr_neutral_evolution/DHFR/contact_map.npy"

run_plmdca_analysis_dhfr(
    input_fasta=input_fasta,
    contacts_file=contacts_file,
    output_root=output_root,
    file_model_scores=model_scores
)

