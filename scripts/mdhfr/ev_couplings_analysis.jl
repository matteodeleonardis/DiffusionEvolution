import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution

in_nat = ARGS[1]
in_wt = ARGS[2]
in_r1 = ARGS[3]
in_r2 = ARGS[4]
in_r3 = ARGS[5]
in_r4 = ARGS[6]
in_r5 = ARGS[7]
in_r15 = ARGS[8]
evc_score_dir = ARGS[9]
output_root = ARGS[10]
model_scores = ARGS[11:end]
contacts_file = joinpath(@__DIR__, "..", "..", "data/dhfr/contact_map.npy")

run_evcouplings_analysis_dhfr(
    in_nat=in_nat,
    in_wt=in_wt,
    in_r1=in_r1,
    in_r2=in_r2,
    in_r3=in_r3,
    in_r4=in_r4,
    in_r5=in_r5,
    in_r15=in_r15,
    evc_score_dir=evc_score_dir,
    contacts_file=contacts_file,
    output_root=output_root,
    file_model_scores=model_scores
)

