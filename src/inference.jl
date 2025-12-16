function infer_series(x, data, times=[]; lambda=0.0, epsilon_J=0.0, epsilon_sigma=0.0, symmetrize=true)
    
    J = compute_J(x, data.d, epsilon_J)
    theta = compute_theta(x, data.d)
    gamma = get_gamma(x, data.d)

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


function get_params_tens(x, pca, d, epsilon_J, A, L; eps_warning=1.0e-4, set_zero=false)

    W_proj = Matrix(pca.proj[:,1:d]')
    J_tens, h_tens, gamma = get_potts_params(x, W_proj, pca.mean, d=d, epsilon=epsilon_J, A=A, L=L, eps_warn=eps_warning, set_zero=set_zero)

    return J_tens, h_tens, gamma
end






