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
    logdet_sigma = 2.0 * sum(log, diag(C.L))
    x_μ = data.round[t].x .- μ
    inv_Σ_x = C \ x_μ
    lls = -0.5*(logdet_sigma + data.d*log2pi) .- vec(0.5*sum(x_μ .* inv_Σ_x, dims=1))

    return lls ./ data.d
end


function empirical_log_likelihood(data::Data, t::Int)

    μ = dropdims(mean(data.round[t].x[1:data.d, :], Weights(data.round[t].w), dims=2), dims=2)
    Σ = cov(data.round[t].x[1:data.d, :], Weights(data.round[t].w), 2)

    C = cholesky(Σ)
    logdet_sigma = 2.0 * sum(log, diag(C.L))

    x_μ = data.round[t].x .- μ
    inv_Σ_x = C \ x_μ
    lls = -0.5*(logdet_sigma + data.d*log2pi) .- vec(0.5*sum(x_μ .* inv_Σ_x, dims=1))

    return lls ./ data.d
end


function transition_probability(x_opt, data_parent, data_child, child, t_parent, t_child, lambda, epsilon_J, epsilon_sigma;
    normalize_child=false)

    @assert data_parent.d == data_child.d

    J = compute_J(x_opt, data_parent.d, epsilon_J)
    theta = compute_theta(x_opt, data_parent.d)
    gamma = get_gamma(x_opt, data_parent.d)
    delta_t = data_child.time[t_child] - data_parent.time[t_parent]
    lambda_t = compute_lambda(J, gamma, delta_t)
    sigma = compute_sigma(J, lambda_t, data_parent.d)
    sigma = (1.0 - lambda)*sigma + lambda*I(data_parent.d)
    sigma = 0.5 * (sigma + sigma')
    sigma += epsilon_sigma*I(data_parent.d)
    C = cholesky(sigma)
    logdet_sigma = 2.0 * sum(log, diag(C.L))

    max_logp = fill(-Inf, length(child))
    for ip in 1:data_parent.M
        if data_parent.round[t_parent].w[ip] == 0.0
            continue
        end
        mu = compute_mu( data_parent.round[t_parent].x[:, ip], lambda_t, theta, data_parent.d)
        x_μ = data_child.round[t_child].x[:, child] .- mu
        inv_sigma_x = C \ x_μ
        logp = -0.5*(logdet_sigma + data_parent.d*log2pi) .- vec(0.5.*sum(x_μ .* inv_sigma_x, dims=1)) .+ log(data_parent.round[t_parent].w[ip])
        logp ./= data_parent.d
        for i in eachindex(child)
            if logp[i] > max_logp[i]
                max_logp[i] = logp[i]
            end
        end
    end

    if normalize_child
        max_logp .-= log(data_parent.round[t_parent].w[child])
    end

    @assert all(isfinite.(max_logp)) "$(max_logp)"
    return max_logp
end






