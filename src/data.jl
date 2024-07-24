struct Sample
    x::Matrix{Float64}
    w::Vector{Float64}
end

struct Data

    x0::Vector{Float64}
    round::Vector{Sample}
    time::Vector{Int} #[t1, ..., tN] we assume t0=0
    M::Int #number of samples
    d::Int
end

function collect_data(x0::Vector{Float64}, coordinates::Array{Float64, 3}, counts::Matrix, time::Vector{Int})

    @assert size(counts,2) == length(time) 

    w = Float64.(counts)
    if !prod(sum(w, dims=1) .≈ 1.0)
        w ./= sum(w, dims=1)
    end

    sample = Vector{Sample}(undef, size(counts, 2))
    for t in axes(coordinates, 3)
        sample[t] = Sample(coordinates[:,:,t], w[:,t])
    end

    

    return Data(x0, sample, time, size(w,2), size(coordinates, 1))
end