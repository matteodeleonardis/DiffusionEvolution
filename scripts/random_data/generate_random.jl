import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution

fasta_wt = ARGS[1]
p_mut = parse(Float64, ARGS[2])
n_rounds = parse(Int, ARGS[3])
n_seqs = parse(Int, ARGS[4])
output_root = ARGS[5]

generate_random_data(
    fasta_wt,
    p_mut,
    n_rounds,
    n_seqs;
    output_root=output_root
)