module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt

    const Pars = Vector{Float64}

    include("data.jl")
    export collect_data, Data
    include("parameters.jl")
    include("learn.jl")
    export learn
    include("utils.jl")

end
