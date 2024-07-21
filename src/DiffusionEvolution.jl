module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim, ProgressMeter

    const Pars = Vector{Float64}

    include("data.jl")
    export collect_data, Data
    include("parameters.jl")
    include("learn.jl")
    export learn_nlopt, learn_gamma_nlopt, learn_gd
    include("utils.jl")

    #simulation
    include("simulate/simulate_ou.jl")
    export random_pars, simulate_ou_process

end
