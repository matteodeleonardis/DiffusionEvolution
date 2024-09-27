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


function which_par(i::Int, d)
    if i == npars_gamma(d)
        return (:gamma, 0)
    elseif i > n_couplings(d)
        return (:field, i - n_couplings(d))
    else
        j_minus_1 = (i ÷ d)
        return (:coupling, j_minus_1 + 1, i - j_minus_1*d)
    end
end

        
function compute_J(x::Pars, d::Int, ϵ::Float64)

    m = [get_Jparameter(x, i, j,d) for i in 1:d, j in 1:d]
    J = m * m'

    return J + ϵ*I(d)
end


function compute_theta(x::Pars, d::Int)

    θ = [get_Hparameter(x, i, d) for i in 1:d]

    return θ
end


function compute_lambda(J, γ::Float64, t)

    return exp(-γ*t*J)
end


function compute_mu(x0::Vector{Float64}, Λt, θ::Vector{Float64}, d::Int)

    return Λt*x0 .+ (I(d)-Λt)*θ  
end


function compute_sigma(J, Λt, d::Int)

    return inv(J)*(I(d) - Λt^2)
end


function compute_parameters(x::Pars, γ::Float64, t::Int, x0::Vector{Float64}, d::Int,
    ϵ::Float64, λ::Float64)

    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)
    Λt = compute_lambda(J, γ, t)
    μ = compute_mu(x0, Λt, θ, d)
    Σ = (1.0-λ)*compute_sigma(J, Λt, d)
    Σ += λ*I(d)

    return μ, Σ
end


function compute_parameters(x::Pars, t::Int, x0::Vector{Float64}, d::Int, 
    ϵ::Float64, λ::Float64)

    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)
    Λt = compute_lambda(J, x[gamma_index(d)], t)
    μ = compute_mu(x0, Λt, θ, d)
    Σ = (1.0-λ)*compute_sigma(J, Λt, d)
    Σ += λ*I(d)

    return μ, Σ
end


function compute_mu_small_gamma(x0::Vector{Float64}, J::Matrix{Float64}, θ::Vector{Float64}, 
    γ::Float64, t::Int)

    return x0 + γ*t*J*(θ-x0)
end


function compute_sigma_small_gamma(J::Matrix{Float64}, γ::Float64, t::Int, d::Int)

    return inv(2.0*γ*t)*I(d)+0.5*J
end


function compute_parameters_small_gamma(x::Pars, γ::Float64, t::Int, x0::Vector{Float64}, d::Int,
    ϵ::Float64, λ::Float64)

    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)
    μ = compute_mu_small_gamma(x0, J, θ, γ, t)
    invΣ = (1.0-λ)*compute_sigma_small_gamma(J, γ, t, d)
    invΣ += λ*I(d)

    return μ, invΣ
end


function compute_parameters_small_gamma(x::Pars, t::Int, x0::Vector{Float64}, d::Int, 
    ϵ::Float64, λ::Float64)

    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)
    μ = compute_mu_small_gamma(x0, J, θ, x[gamma_index(d)], t)
    invΣ = (1.0-λ)*compute_sigma_small_gamma(J, x[gamma_index(d)], t, d)
    invΣ += λ*I(d)

    return μ, invΣ
end