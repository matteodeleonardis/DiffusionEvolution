import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using DiffusionEvolution

d = parse(Int, ARGS[1])
output_root = ARGS[2]
contacts_file = joinpath(@__DIR__, "../../data/dhfr/contact_map.npy")

run_analysis_dhfr(
    d=d, 
    opt_pkg=:Optim, 
    output_root=output_root,
    contacts_file=contacts_file
)