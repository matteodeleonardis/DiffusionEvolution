import Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))

using DiffusionEvolution

file_wt = joinpath(@__DIR__, "../../data/pse1/PSE1.fas")
file_round1 = joinpath(@__DIR__, "../../data/pse1/Rnd10.fas")
file_round2 = joinpath(@__DIR__, "../../data/pse1/Rnd20_init.fas")
file_nat = joinpath(@__DIR__, "../../data/pse1/PSE1_clean.fasta")

model_pars = ARGS

rand_seq_analysis_pse1(
    file_wt, 
    [file_round1, file_round2], 
    file_nat, 
    ARGS
)