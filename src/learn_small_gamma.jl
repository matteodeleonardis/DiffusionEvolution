function log_likelihood_small_gamma(x::Pars,  data::Data, prior_x::Float64, prior_γ::Float64, ϵ::Float64)

    ll = 0.0
    for t in eachindex(data.round)
        μ, invΣ = compute_parameters_small_gamma(x, data.time[t], data.x0, data.d, ϵ)
        ll += weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), invΣ)
        ll += data.d*log2pi - logdet(invΣ)
    end

    if prior_x > 0.0
        for i in 1:length(x)-1
            ll += prior_x*(x[i]^2)
        end
    end
    if prior_γ > 0.0
        ll += prior_γ*(x[end]^2)
    end

    return  ll
end


function optim_wrapper_small_gamma(x::Pars, g::Pars, data::Data, prior_x::Float64, prior_γ::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood_small_gamma(par, data, prior_x, prior_γ, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_small_gamma_nlopt(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1,
    prior_x=0.0, prior_gamma=0.0, epsilon=0.0)

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

    opt.min_objective = (x,g) -> optim_wrapper_small_gamma(x, g, data, prior_x, prior_gamma, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
        x0[end] = 1e-3
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true)
        x0[end] = 1e-3
    end

    x_start = copy(x0)

    (minf, minx, status) = NLopt.optimize!(opt, x0)

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end


function optimize_small_gamma_gd!(data::Data; x=randn(npars_gamma(data.d)), initialize=-1, 
    prior_x=0.0, prior_gamma=0.0, epsilon=0.0, 
    eta=0.001, iterations=1, verbose=false)

    ll_iter = fill(+Inf, iterations+1)

    #setting initial condition for x
    if initialize == 0
        init_id!(x, d=data.d, init_gamma=true)
        x0[end] = 1e-3
    elseif initialize>0
        init_cov!(x, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true)
        x0[end] = 1e-3
    end

    ll_iter[1] = log_likelihood_small_gamma(x, data, prior_x, prior_gamma, epsilon)
    g_x = zeros(npars_gamma(data.d))
    x_update = zeros(npars_gamma(data.d))
    for it in 1:iterations
        if verbose
            println("iteration $it/$iterations")
        end
        optim_wrapper_small_gamma(x, g_x, data, prior_x, prior_gamma, epsilon)
        x_update .= (x .- (eta * g_x))
        ll_iter[it+1] = log_likelihood_small_gamma(x_update, data, prior_x, prior_gamma, epsilon)
        if ll_iter[it+1] <= ll_iter[it]
            x .= x_update
        else
            println("Log-likelihood has increased. Optimization stopped.")
            break
        end
    end

    return ll_iter
end