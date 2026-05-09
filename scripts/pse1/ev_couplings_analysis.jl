import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution


in_nat = ARGS[1]
in_wt = ARGS[2]
in_r10 = ARGS[3]
in_r20 = ARGS[4]
evc_score_dir = ARGS[5]
output_root = ARGS[6]
model_scores = ARGS[7:end]
contacts_file = joinpath(@__DIR__, "../../data/pse1/contact_map.jld2")

run_evcouplings_analysis_pse1(
    in_nat=in_nat,
    in_wt=in_wt,
    in_r10=in_r10,
    in_r20=in_r20,
    evc_score_dir=evc_score_dir,
    contacts_file=contacts_file,
    output_root=output_root,
    file_model_scores=model_scores
)

