function loglikelihood_noalloc(x::Vector{Float64}, data::Data, Jcheck; d::Int)

    ll = 0.0
    Jtri = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    J = Jtri .+ Jtri'
    @assert maximum(abs.(J .- Jcheck)) < 1e-15
    Λ = exp(-J) 
     
    for t in eachindex(data.round)
        μ = (Λ^data.delta[t]) * data.round[t].x
        Σ = inv(J) * (I(d) - Λ^(2*data.delta[t]))
        ll += log(det(Σ)) #+ weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), inv(Σ))
    end

    return ll
end