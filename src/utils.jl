function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d)

    C = cov(Xdata, Weights(w), 2)
    @assert issymmetric(C)
    J = inv(C)

    for i in 1:d
        for j in i:d
            x0[index(i,j)] = J[i,j]
        end
    end
end