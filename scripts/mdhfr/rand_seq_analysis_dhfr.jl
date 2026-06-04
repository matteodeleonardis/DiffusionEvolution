import Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))

using DiffusionEvolution

file_wt = joinpath(@__DIR__, "../../data/dhfr/mDHFR.fasta")
file_round1 = joinpath(@__DIR__, "../../data/dhfr/Round1_Q15_C10_aa.aln")
file_round2 = joinpath(@__DIR__, "../../data/dhfr/Round2_Q15_C10_aa.aln")
file_round3 = joinpath(@__DIR__, "../../data/dhfr/Round3_Q15_C10_aa.aln")
file_round4 = joinpath(@__DIR__, "../../data/dhfr/Round4_Q15_C10_aa.aln")
file_round5 = joinpath(@__DIR__, "../../data/dhfr/Round5_Q15_C10_aa.aln")
file_round15 = joinpath(@__DIR__, "../../data/dhfr/Gen15_aa.aln")
file_nat = joinpath(@__DIR__, "../../data/dhfr/mDHFR_clean.fasta")

model_pars = ARGS

rand_seq_analysis_dhfr(
    file_wt, 
    [file_round1, file_round2, file_round3, file_round4, file_round5, file_round15], 
    file_nat, 
    ARGS
)