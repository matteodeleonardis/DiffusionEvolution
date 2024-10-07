function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d, init_gamma = -1.0)

    println("Initializing parameters with covariance.")
    m = mean(Xdata, Weights(w), dims=2)
    C = cov(Xdata, Weights(w), 2)
    
    @assert isapprox(C,C')
    J = svd_inv(C)
    x_s = sqrt(J)


    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = x_s[i,j]
        end
        x0[Hindex(i, d)] = m[i]
    end  
    
    if init_gamma > 0.0
        x0[gamma_index(d)] = init_gamma
    end
end

function init_id!(x0::Pars; d, init_gamma = -1.0)

    println("Initializing parameters with identity.")
    J = I(d)

    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = J[i,j]
        end
        x0[Hindex(i, d)] = 0.0
    end  
    
    if init_gamma > 0.0
        x0[gamma_index(d)] = init_gamma
    end
end


function compute_energy(x_c::Matrix{Float64}, J::Matrix{Float64}, θ::Vector{Float64})
    
    xm = (x_c .- θ)
    return vec(sum(xm .* (J*xm), dims=1))
end

function compute_energy(x_c::Matrix{Float64}, x::Vector{Float64}, ϵ::Float64)

    d = size(x_c, 1)
    J = compute_J(x, d, ϵ)*x[end]
    θ = compute_theta(x, d)

    return compute_energy(x_c, J, θ)
end


function compute_weight(x_c::Matrix{Float64}, x::Vector{Float64}, γ, t::Int, x0::Vector{Float64},
    ϵ::Float64, λ::Float64)

    μ, Σ = compute_parameters(x, γ, t, x0, size(x_c, 1), ϵ, λ)
    return compute_energy(x_c, inv(Σ), μ)
end


function safe_log(x::Float64, ϵ::Float64)

    return log(x + ϵ)
end


function get_potts_params(x::Pars, Wproj::Matrix{Float64}, x_mean::Vector{Float64}; d::Int, epsilon::Float64, 
    A::Int, L::Int, eps_warn = 1.0e-4, set_zero=false)

    @assert size(Wproj, 1) <= size(Wproj, 2) #projects on a smaller space
    @assert L*A == size(Wproj, 2)

    J_embedding = compute_J(x, d, epsilon)
    if length(x) == npars_gamma(d)
        J_embedding .*= x[end]
    end
    θ_embedding = compute_theta(x, d)
    J_potts = -(Wproj')*J_embedding*Wproj
    h_potts = vec(2 * ((Wproj*x_mean)' + θ_embedding') * J_embedding * Wproj)
    h_potts_tens = reshape(h_potts, A, L)
    J_potts_tens = permutedims(reshape(J_potts, A, L, A, L), (1,3,2,4))
    flag_warning = false

    for i in axes(J_potts_tens, 3)
        for a in 1:A
            h_potts_tens[a,i] += J_potts_tens[a,a,i,i]
            J_potts_tens[a,a,i,i] = 0.0
        end

        if set_zero
            if (!flag_warning) && (maximum(abs.(J_potts_tens[:,:,i,i])) > eps_warn)
                println("Warning: possibly large value neglected in couplings J (>= $(eps_warn))")
                flag_warning = true
            end
            J_potts_tens[:,:,i,i] .= 0.0
        end
    end
    
    γ = -1.0
    if length(x)==npars_gamma(d)
        γ = x[end]
    end

    return (J_potts_tens, h_potts_tens, γ)
end


function svd_inv(m::Matrix{Float64})
    dec = svd(m)
    return dec.V*diagm(inv.(dec.S))*transpose(dec.U)
end