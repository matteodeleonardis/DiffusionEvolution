struct Sample
    x::Matrix{Float64}
    w::Vector{Float64}
end

struct Data

    round::Vector{Sample}
    delta::Vector{Int} #[t1-t0, t2-t1, ..., tN-t(N-1)]
    M::Int #number of samples
end

function collect_data(coordinates::Array{Float64, 3}, counts::Matrix, delta::Vector{Int})

    w = Float64.(counts)
    if !prod(sum(w, dims=1) .≈ 1.0)
        w ./= sum(w, dims=1)
    end

    sample = Vector{Sample}(undef, size(counts, 2))
    for t in axes(coordinates, 3)
        sample[t] = Sample(coordinates[:,:,t], w[:,t])
    end

    

    return Data(sample, delta, size(w,2))
end