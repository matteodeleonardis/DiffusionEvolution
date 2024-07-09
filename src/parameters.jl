function npars(d::Int)

    return (d^2-d)÷2
end


function index(i::Int, j::Int) #i<j

    return (j-1)*(j-2)÷2 + i 
end


function get_parameter(x::Pars, i::Int, j::Int)
    p = 0.0
    if i<j
        p = x[index(i,j)]
    elseif j<i
        p = x[index(j,i)]
    end

    return p
end


function compute_J(x::Pars, d::Int)

    J = [get_parameter(x, i, j) for i in 1:d, j in 1:d]

    return J
end


function compute_lambda(J::Matrix{Float64})

    return exp(-J)
end


function compute_mu(t::Int, data::Data, Λ::Matrix{Float64})

    return (Λ^data.delta[t]) * data.round[t].x
end


function compute_sigma(t::Int, data::Data, J::Matrix{Float64}, Λ::Matrix{Float64})

    return inv(J)*(I(size(J, 1)) - Λ^(2*data.delta[t]))
end


function compute_parameters(x::Pars, t::Int, data::Data, d::Int)

    J = compute_J(x, d)
    Λ = compute_lambda(J)
    μ = compute_mu(t, data, Λ)
    Σ = compute_sigma(t, data, J, Λ)

    return μ, Σ
end