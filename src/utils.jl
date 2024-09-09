function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d, init_gamma = false)

    println("Initializing parameters with covariance.")
    m = mean(Xdata, Weights(w), dims=2)
    C = cov(Xdata, Weights(w), 2)
    
    @assert isapprox(C,C')
    J = inv(C)


    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = J[i,j]
        end
        x0[Hindex(i, d)] = m[i]
    end  
    
    if init_gamma
        x0[gamma_index(d)] = 1.0
    end
end

function init_id!(x0::Pars; d, init_gamma = false)

    println("Initializing parameters with identity.")
    J = I(d)

    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = J[i,j]
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

function compute_energy(x_c::Matrix{Float64}, x::Vector{Float64}, ϵ::Float64)

    d = size(x_c, 1)
    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)

    return compute_energy(x_c, J, θ)
end


function compute_weight(x_c::Matrix{Float64}, x::Vector{Float64}, t::Int, x0::Vector{Float64},
    ϵ::Float64, λ::Float64)

    μ, Σ = compute_parameters(x, t, x0, size(x_c, 1), ϵ, λ)
    return compute_energy(x_c, inv(Σ), μ)
end


function safe_log(x::Float64, ϵ::Float64)

    return log(x + ϵ)
end