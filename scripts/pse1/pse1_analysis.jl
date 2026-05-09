import Pkg
Pkg.activate(joinpath("@__DIR__", "..", ".."))

using DiffusionEvolution

d = parse(Int, ARGS[1])
output_root = ARGS[2]
contacts_file = joinpath(@__DIR__, "../../data/pse1/contact_map.jld2")

run_analysis_pse1(
    d=d, 
    opt_pkg=:Optim, 
    output_root=output_root,
    contacts_file=contacts_file
)