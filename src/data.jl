struct Sample
    x::Matrix{Float64}
end

struct Data

    round::Vector{Sample}
    w::Matrix{Float64}
end

function collect_data(coordinates::Array{Float64, 3}, counts::Matrix)

    sample = Vector{Sample}(undef, size(counts, 2))
    for t in axes(coordinates, 3)
        sample[t] = Sample(coordinates[:,:,t])
    end

    w = Float64.(counts)
    if !prod(sum(w, dims=1) .≈ 1.0)
        w ./= sum(w, dims=1)
    end

    return Data(sample, w)
end