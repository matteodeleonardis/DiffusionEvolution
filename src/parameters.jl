function n_couplings(d::Int)

    return d*d
end


function npars(d::Int)

    n_couplings(d) + d
end


function npars_gamma(d::Int)

    n_couplings(d) + d + 1
end


function Jindex(i::Int, j::Int, d::Int)

    return d*(j-1) + i 
end


function Hindex(i::Int, d::Int)

    return n_couplings(d) + i
end


function gamma_index(d::Int)

    return n_couplings(d) + d + 1
end


function get_Jparameter(x::Pars, i::Int, j::Int, d::Int)

    return x[Jindex(i,j,d)]
end


function get_Hparameter(x::Pars, i::Int, d::Int)

    return x[Hindex(i, d)]
end


function get_gamma(x::Pars, d::Int)

    return x[gamma_index(d)]
end


function compute_J(x::Pars, d::Int)

    m = [get_Jparameter(x, i, j,d) for i in 1:d, j in 1:d]
    J = m * m'

    return J
end


function compute_theta(x::Pars, d::Int)

    θ = [get_Hparameter(x, i, d) for i in 1:d]

    return θ
end


function compute_lambda(J, γ::Float64)

    return exp(-γ*J)
end


function compute_mu(t::Int, x0::Vector{Float64}, Λ, θ::Vector{Float64}, d::Int)

    Λt = Λ^t
    return Λt * x0 .+ (I(d)-Λt)*θ  
end


function compute_sigma(t::Int, J, Λ, d::Int)

    return inv(J)*(I(d) - Λ^(2*t))
end


function compute_parameters(x::Pars, γ::Float64, t::Int, x0::Vector{Float64}, d::Int)

    J = compute_J(x, d)
    θ = compute_theta(x, d)
    Λ = compute_lambda(J, γ)
    μ = compute_mu(t, x0, Λ, θ, d)
    Σ = compute_sigma(t, J, Λ, d)

    return μ, Σ
end


function compute_parameters(x::Pars, t::Int, x0::Vector{Float64}, d::Int)

    J = compute_J(x, d)
    θ = compute_theta(x, d)
    Λ = compute_lambda(J, x[gamma_index(d)])
    μ = compute_mu(t, x0, Λ, θ, d)
    Σ = compute_sigma(t, J, Λ, d)

    return μ, Σ
end