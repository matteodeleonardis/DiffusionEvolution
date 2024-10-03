module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim, Distributions, JLD2

    const Pars = Vector{Float64}
    const log2pi = log(2.0*π)

    include("data.jl")
    export collect_data, Data

    include("parameters.jl")
    export which_par

    include("learn.jl")
    export learn_nlopt, iterative_maximization, optimize_pars_gd!

    include("learn_gamma.jl")
    export learn_gamma_nlopt, learn_gamma_optim, learn_gamma_unconstrained_optim, optimize_gd!

    include("learn_small_gamma.jl")
    export learn_small_gamma_nlopt, optimize_small_gamma_gd!, learn_small_gamma_optim

    include("utils.jl")
    export compute_energy, compute_weight, get_potts_params

    #simulation
    include("simulate/simulate_ou.jl")
    export random_pars, simulate_ou_process

end
