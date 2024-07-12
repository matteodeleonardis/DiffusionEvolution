module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim

    const Pars = Vector{Float64}

    include("data.jl")
    export collect_data, Data
    include("parameters.jl")
    include("learn.jl")
    export learn_nlopt, learn_gd
    include("utils.jl")

end
