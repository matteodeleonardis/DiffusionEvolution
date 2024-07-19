function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d)

    m = mean(Xdata, dims=2)
    Δ = Xdata[:,:,end] .- m
    C = Δ * (reshape(w, :, 1) .* Δ')
    @assert isapprox(C,C')
    J = inv(C)

    for i in 1:d
        for j in i:d
            x0[Jindex(i,j)] = J[i,j]
        end
        x0[Hindex(i, d)] = m[i]
    end  
    #x0[gamma_index(d)] = 100.0  
end