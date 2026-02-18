import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution

input_fasta = ARGS[1]
plmdca_dir = ARGS[2]
output_root = ARGS[3]
gamma = ARGS[4] == "no"
model_scores = ARGS[5:end]
contacts_file = "/home/students/s301803/dhfr_neutral_evolution/DHFR/contact_map.npy"

run_plmdca_analysis_dhfr(
    input_fasta=input_fasta,
    plmdca_dir=plmdca_dir,
    contacts_file=contacts_file,
    output_root=output_root,
    fixed = gamma,
    file_model_scores=model_scores
)

