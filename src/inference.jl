function infer_series(x, data, times=[]; lambda=0.0, epsilon_J=0.0, epsilon_sigma=0.0, symmetrize=true)
    
    J = compute_J(x, data.d, epsilon_J)
    theta = compute_theta(x, data.d)
    if length(x) == npars_gamma(data.d)
        gamma = get_gamma(x, data.d)
    else
        gamma = 1.0
    end

    sigma = Dict{Int, Matrix{Float64}}()
    mu = Dict{Int, Vector{Float64}}()

    if length(times) == 0
        times = eachindex(data.round)
    end

    for t in times
        lambda_t = compute_lambda(J, gamma, t)
        mu[t] = compute_mu(data.x0, lambda_t, theta, data.d)
        _sigma = (1.0 - lambda)*compute_sigma(J, lambda_t, data.d)
        _sigma += lambda*I(data.d)

        if symmetrize
            _sigma = 0.5 * (_sigma + _sigma')
        end

        _sigma += epsilon_sigma*I(data.d)
        sigma[t] = _sigma
    end

    C_J = cholesky(J)
    sigma_eq = C_J \ I(data.d)
    if symmetrize
        sigma_eq = 0.5 * (sigma_eq + sigma_eq')
    end
    sigma_eq += epsilon_sigma*I(data.d)

    return mu, sigma, theta, sigma_eq

end


function fit_series(data, times=[])

    if length(times) == 0
        times = eachindex(data.round)
    end

    sigma = Dict{Int, Matrix{Float64}}()
    mu = Dict{Int, Vector{Float64}}()

    for t in times
        mu[t] = dropdims(mean(data.round[t].x[1:data.d, :], Weights(data.round[t].w), dims=2), dims=2)
        sigma[t] = cov(data.round[t].x[1:data.d, :], Weights(data.round[t].w), 2)
    end

    return mu, sigma
end


function get_params_tens(x, pca, d, epsilon_J, A, L; whiten, epsilon_rel=1.0e-8, eps_warning=1.0e-4, set_zero=false)

    W_proj = Matrix(pca.proj[:,1:d]')
    if whiten
        lambda = principalvars(pca)
        epsilon = epsilon_rel * maximum(lambda)
        W_proj = (1.0 ./ sqrt.(lambda .+ epsilon)) .* W_proj
    end
    J_tens, h_tens, gamma = get_potts_params(x, W_proj, pca.mean, d=d, epsilon=epsilon_J, A=A, L=L, eps_warn=eps_warning, set_zero=set_zero)

    return J_tens, h_tens, gamma
end


function estimate_gamma(data)

    mu, sigma = fit_series(data)
    sigma_traces = [tr(sigma[t]) for t in eachindex(data.time)]

    return sum(sigma_traces)/(2.0*data.d*sum(data.time))
end


function log_likelihood_variants(x::Pars,  data::Data, t::Int, λ::Float64, ϵ_J::Float64, ϵ_Σ::Float64)

    J = compute_J(x, data.d, ϵ_J)
    θ = compute_theta(x, data.d)
    γ = get_gamma(x, data.d)
   
    μ, Σ = compute_parameters(J, θ, γ, data.time[t], data.x0, data.d, λ, ϵ_Σ)
    C = cholesky(Σ)
    x_μ = data.round[t].x .- μ
    inv_Σ_x = C \ x_μ
    lls = -sum(x_μ .* inv_Σ_x, dims=1)/data.d

    return vec(lls)
end


function empirical_log_likelihood(data::Data, t::Int)

    μ = dropdims(mean(data.round[t].x[1:data.d, :], Weights(data.round[t].w), dims=2), dims=2)
    Σ = cov(data.round[t].x[1:data.d, :], Weights(data.round[t].w), 2)

    C = cholesky(Σ)
    x_μ = data.round[t].x .- μ
    inv_Σ_x = C \ x_μ
    lls = -sum(x_μ .* inv_Σ_x, dims=1)/data.d

    return vec(lls)
end






