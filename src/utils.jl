function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d, init_gamma = false, rescale=true)

    m = mean(Xdata, dims=2)
    Δ = Xdata[:,:,end] .- m
    C = Δ * (reshape(w, :, 1) .* Δ')
    if rescale 
        C ./= sqrt.(diag(C)) * sqrt.(diag(C))'
    end
    @assert isapprox(C,C')
    J = inv(C)


    for i in 1:d
        for j in i:d
            x0[Jindex(i,j)] = J[i,j]
        end
        x0[Hindex(i, d)] = m[i]
    end  
    
    if init_gamma
        x0[gamma_index(d)] = 1.0
    end
end

function init_id!(x0::Pars; d, init_gamma = false)

    J = I(d)

    for i in 1:d
        for j in i:d
            x0[Jindex(i,j)] = J[i,j]
        end
        x0[Hindex(i, d)] = 0.0
    end  
    
    if init_gamma
        x0[gamma_index(d)] = 1.0
    end
end


function compute_energy(x_c::Matrix{Float64}, J::Matrix{Float64}, θ::Vector{Float64})
    
    xm = (x_c .- θ)
    return vec(sum(xm .* (J*xm), dims=1))
end

function compute_energy(x_c::Matrix{Float64}, x::Vector{Float64})

    d = size(x_c, 1)
    J = compute_J(x, d)
    θ = compute_theta(x, d)

    return compute_energy(x_c, J, θ)
end