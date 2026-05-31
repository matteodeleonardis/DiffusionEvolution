function log_likelihood_gamma(x::Pars,  data::Data, λ::Float64, prior_J::Float64, prior_theta::Float64, 
    prior_γ::Float64, ϵ_J::Float64, ϵ_Σ::Float64)

    ll = 0.0
    J = compute_J(x, data.d, ϵ_J)
    θ = compute_theta(x, data.d)
    γ = get_gamma(x, data.d)
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(J, θ, γ, data.time[t], data.x0, data.d, λ, ϵ_Σ)
        C = cholesky(Σ)
        x_μ = data.round[t].x .- μ
        inv_Σ_x = C \ x_μ
        mahalanobis = 0.5 .* vec(sum(x_μ .* inv_Σ_x, dims=1))
        ll += dot(data.round[t].w, mahalanobis)/data.d + logsumexp(-mahalanobis)/data.d
    end


    if prior_J > 0.0
        ll += prior_J*sum(abs2, J)/data.d^2
    end
    if prior_theta > 0.0
        ll += prior_theta*sum(abs2, θ)/data.d
    end
    if prior_γ > 0.0
        ll += prior_γ*abs2(log1pexp(-x[gamma_index(data.d)])) #it is -log(gamma) since gamma is 1/(1+exp(-x[gamma_index(data.d)]))
    end

    return  ll
end


function log_likelihood_fixed(x::Pars,  data::Data, λ::Float64, prior_J::Float64, prior_theta::Float64, 
    ϵ_J::Float64, ϵ_Σ::Float64)

    ll = 0.0
    J = compute_J(x, data.d, ϵ_J)
    θ = compute_theta(x, data.d)
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(J, θ, 1.0, data.time[t], data.x0, data.d, λ, ϵ_Σ)
        C = cholesky(Σ)
        ll += (2*sum(log, diag(C.U)) + data.d*log2pi)/data.d
        x_μ = data.round[t].x .- μ
        inv_Σ_x = C \ x_μ
        ll += sum((data.round[t].w' .* x_μ) .* inv_Σ_x)/data.d
    end


    if prior_J > 0.0
        ll += prior_J*sum(abs2, J)/data.d^2
    end
    if prior_theta > 0.0
        ll += prior_theta*sum(abs2, θ)/data.d
    end

    return  ll
end


function optim_wrapper_gamma(x::Pars, g::Pars, data::Data, λ::Float64, prior_J::Float64, 
    prior_theta::Float64, prior_γ::Float64, ϵ_J::Float64, ϵ_Σ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = Flux.gradient(x) do par
        ll = log_likelihood_gamma(par, data, λ, prior_J, prior_theta, prior_γ, ϵ_J, ϵ_Σ)
    end

    g .= gs[1]
    return ll
end


function optim_wrapper_fixed(x::Pars, g::Pars, data::Data, λ::Float64, prior_J::Float64, 
    prior_theta::Float64, ϵ_J::Float64, ϵ_Σ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = Flux.gradient(x) do par
        ll = log_likelihood_fixed(par, data, λ, prior_J, prior_theta, ϵ_J, ϵ_Σ)
    end

    g .= gs[1]
    return ll
end


function learn_gamma_nlopt(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon_J=0.0, epsilon_sigma=0.0)

    opt = Opt(alg, npars_gamma(data.d))
    lb = fill(-Inf, npars_gamma(data.d))
    lb[gamma_index(data.d)] = 0.0
    opt.lower_bounds = lb
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper_gamma(x, g, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=#=inv(data.time[end])=#1.0e-5)
    end

    x_start = copy(x0)

    (minf, minx, status) = NLopt.optimize!(opt, x0)

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end


function learn_gamma_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, 
    epsilon_J=0.0, epsilon_sigma=0.0, stop_tol...)

    x_gamma_0 = logit(inv(data.time[end]))

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=x_gamma_0)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = optim_wrapper_gamma(x, G, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
        elseif F!== nothing
            ll = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
        end

        return ll
    end

    println("*** Gamma initalization ***")
    println("x[gamma]: ", x0[gamma_index(data.d)])
    println("gamma value: ", get_gamma(x0, data.d))
    println()

    res = Optim.optimize(NLSolversBase.only_fg!(fg!), x0, alg, Optim.Options(; stop_tol...))

    return res
end


function learn_fixed_optim(data::Data; x0=randn(npars(data.d)), initialize=-1,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, 
    epsilon_J=0.0, epsilon_sigma=0.0, stop_tol...)

    if initialize == 0
        init_id!(x0, d=data.d)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = optim_wrapper_fixed(x, G, data, lambda, prior_J, prior_theta, epsilon_J, epsilon_sigma)
        elseif F!== nothing
            ll = log_likelihood_fixed(x, data, lambda, prior_J, prior_theta, epsilon_J, epsilon_sigma)
        end

        return ll
    end

    res = Optim.optimize(NLSolversBase.only_fg!(fg!), x0, alg, Optim.Options(; stop_tol...))

    return res
end


function learn_gamma_unconstrained_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon_J=0.0, epsilon_sigma=0.0, g_tol=1e-8, f_tol=0.0, err_file="err_file")

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll =  optim_wrapper_gamma(x, G, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
        elseif F!== nothing
            ll = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
        end

        return ll
    end

    res = Optim.optimize(NLSolversBase.only_fg!(fg!), x0, alg, Optim.Options(g_tol=g_tol, f_tol=f_tol))

    return res
end


function optimize_gd!(data::Data; x=randn(npars_gamma(data.d)), initialize=-1, 
    lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon_J=0.0, epsilon_sigma=0.0, 
    eta=0.001, iterations=1, verbose=false)

    ll_iter = fill(+Inf, iterations+1)
    g_iter = zeros(iterations+1)

    #setting initial condition for x
    if initialize == 0
        init_id!(x, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=inv(data.time[end]))
    end

    ll_iter[1] = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
    g_x = zeros(npars_gamma(data.d))
    x_update = zeros(npars_gamma(data.d))
    for it in 1:iterations
        if verbose
            println("iteration $it/$iterations")
        end
        optim_wrapper_gamma(x, g_x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
        g_iter[it] = maximum(abs.(g_x))
        x_update .= (x .- (eta * g_x))
        ll_iter[it+1] = log_likelihood_gamma(x_update, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, sigma_sigma)
        if ll_iter[it+1] <= ll_iter[it]
            x .= x_update
        else
            println("Log-likelihood has increased. Optimization stopped.")
            break
        end
    end
    optim_wrapper_gamma(x, g_x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon_J, epsilon_sigma)
    g_iter[end] = maximum(abs.(g_x))

    return ll_iter, g_iter
end