function n_couplings(d::Int)

    return (d^2-d)÷2+d
end


function npars(d::Int)

    n_couplings(d) + d
end


function Jindex(i::Int, j::Int) #i<j

    return j*(j-1)÷2 + i 
end


function Hindex(i::Int, d::Int)

    return n_couplings(d) + i
end


function get_Jparameter(x::Pars, i::Int, j::Int)
    p = 0.0
    if i<=j
        p = x[Jindex(i,j)]
    elseif j<i
        p = x[Jindex(j,i)]
    end

    return p
end


function get_Hparameter(x::Pars, i::Int, d::Int)

    return x[Hindex(i, d)]
end


function compute_J(x::Pars, d::Int)

    J = [get_Jparameter(x, i, j) for i in 1:d, j in 1:d]

    return J
end


function compute_theta(x::Pars, d::Int)

    θ = [get_Hparameter(x, i, d) for i in 1:d]

    return θ
end


function compute_lambda(J::Matrix{Float64})

    return exp(-J)
end


function compute_mu(t::Int, data::Data, Λ::Matrix{Float64}, θ::Vector{Float64}, d::Int)

    Λt = Λ^data.delta[t]
    return Λt * data.round[t].x .+ (θ' * (I(d)-Λt))'
end


function compute_sigma(t::Int, data::Data, J::Matrix{Float64}, Λ::Matrix{Float64})

    return inv(J)*(I(size(J, 1)) - Λ^(2*data.delta[t]))
end


function compute_parameters(x::Pars, t::Int, data::Data, d::Int)

    J = compute_J(x, d)
    θ = compute_theta(x, d)
    Λ = compute_lambda(J)
    μ = compute_mu(t, data, Λ, θ, d)
    Σ = compute_sigma(t, data, J, Λ)

    return μ, Σ
end