function log_likelihood_gamma(x::Pars,  data::Data, λ::Float64, prior_J::Float64, prior_theta::Float64, prior_γ::Float64, ϵ::Float64)

    ll = 0.0
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(x, data.time[t], data.x0, data.d, ϵ, λ)
        ll += logdet(Σ) + weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), svd_inv(Σ))
        ll += data.d*log2pi
    end

    if prior_J > 0.0
        J = compute_J(x, data.d, ϵ)
        ll += prior_J*sum(y->y^2, J)
    end
    if prior_theta > 0.0
        θ = compute_theta(x, data.d)
        ll += prior_theta*sum(y->y^2, θ)
    end
    if prior_γ > 0.0
        ll += prior_γ*(x[end]^2)
    end

    return  ll
end


function optim_wrapper_gamma(x::Pars, g::Pars, data::Data, λ::Float64, prior_J::Float64, prior_theta::Float64, prior_γ::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood_gamma(par, data, λ, prior_J, prior_theta, prior_γ, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_gamma_nlopt(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon=0.0)

    opt = Opt(alg, npars_gamma(data.d))
    lb = fill(-Inf, npars_gamma(data.d))
    lb[gamma_index(data.d)] = 1e-12
    opt.lower_bounds = lb
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper_gamma(x, g, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=inv(data.time[end]))
    end

    x_start = copy(x0)

    (minf, minx, status) = NLopt.optimize!(opt, x0)

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end


function learn_gamma_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon=0.0, g_tol=1e-8, f_tol=0.0, x_tol=0.0)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=inv(data.time[end]))
    end

    lower = vcat(fill(-Inf, npars(data.d)), 0.0)
    upper = fill(+Inf, npars_gamma(data.d))

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = optim_wrapper_gamma(x, G, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        elseif F!== nothing
            ll = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        end

        return ll
    end

    res = Optim.optimize(Optim.only_fg!(fg!), lower, upper, x0, Fminbox(alg), Optim.Options(g_tol=g_tol, f_tol=f_tol, x_tol=x_tol))

    return res
end


function learn_gamma_unconstrained_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon=0.0, g_tol=1e-8, f_tol=0.0, err_file="err_file")

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll =  optim_wrapper_gamma(x, G, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        elseif F!== nothing
            ll = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        end

        return ll
    end

    res = Optim.optimize(Optim.only_fg!(fg!), x0, alg, Optim.Options(g_tol=g_tol, f_tol=f_tol))

    return res
end


function optimize_gd!(data::Data; x=randn(npars_gamma(data.d)), initialize=-1, 
    lambda=0.0, prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon=0.0, 
    eta=0.001, iterations=1, verbose=false)

    ll_iter = fill(+Inf, iterations+1)
    g_iter = zeros(iterations+1)

    #setting initial condition for x
    if initialize == 0
        init_id!(x, d=data.d, init_gamma=1.0)
    elseif initialize>0
        init_cov!(x, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=inv(data.time[end]))
    end

    ll_iter[1] = log_likelihood_gamma(x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
    g_x = zeros(npars_gamma(data.d))
    x_update = zeros(npars_gamma(data.d))
    for it in 1:iterations
        if verbose
            println("iteration $it/$iterations")
        end
        optim_wrapper_gamma(x, g_x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        g_iter[it] = maximum(abs.(g_x))
        x_update .= (x .- (eta * g_x))
        ll_iter[it+1] = log_likelihood_gamma(x_update, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
        if ll_iter[it+1] <= ll_iter[it]
            x .= x_update
        else
            println("Log-likelihood has increased. Optimization stopped.")
            break
        end
    end
    optim_wrapper_gamma(x, g_x, data, lambda, prior_J, prior_theta, prior_gamma, epsilon)
    g_iter[end] = maximum(abs.(g_x))

    return ll_iter, g_iter
end