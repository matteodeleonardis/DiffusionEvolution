module DiffusionEvolution

    using LinearAlgebra

    const Pars = Vector{Float64}

    include("data.jl")
    export collect_data, Data
    include("workspace.jl")
    include("parameters.jl")
    export index, compute_J!, compute_lambda!, compute_sigma!

end
