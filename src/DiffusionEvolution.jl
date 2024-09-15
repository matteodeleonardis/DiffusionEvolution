module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim, Distributions, JLD2

    const Pars = Vector{Float64}
    const log2pi = log(2.0*π)

    include("data.jl")
    export collect_data, Data

    include("parameters.jl")

    include("learn.jl")
    export learn_nlopt, iterative_maximization

    include("learn_gamma.jl")
    export learn_gamma_nlopt, learn_gamma_optim, learn_gamma_unconstrained_optim

    include("utils.jl")
    export compute_energy, compute_weight

    #simulation
    include("simulate/simulate_ou.jl")
    export random_pars, simulate_ou_process

end
