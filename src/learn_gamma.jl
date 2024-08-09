function log_likelihood_gamma(x::Pars,  data::Data, λ::Float64, prior::Float64, ϵ::Float64)

    ll = 0.0
    T = length(data.round)
    for t in 1:T-1
        μ, Σ = compute_parameters(x, data.time[t+1], data.x0, data.d)
        ll += safe_log(det((1.0-λ)*Σ + λ*I(data.d)), ϵ) + weighted_batch_dot(data.round[t+1].w, (data.round[t+1].x .- μ), inv((1.0-λ)*Σ + λ*I(data.d)))
    end

    ll /= length(data.round)
    if prior > 0.0
        for i in 1:length(x)-1
            ll += prior*(x[i]^2)
        end
    end

    return  ll
end


function optim_wrapper_gamma(x::Pars, g::Pars, data::Data, λ::Float64, prior::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood_gamma(par, data, λ, prior, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_gamma_nlopt(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, λ=0.0,
    prior=0.0, rescale=false, epsilon=0.0)

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

    opt.min_objective = (x,g) -> optim_wrapper_gamma(x, g, data, λ, prior, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true, 
        rescale=rescale)
    end

    x_start = copy(x0)

    (minf, minx, status) = try NLopt.optimize!(opt, x0)
    catch e
        println("Optimization failed. \n", e)
        return (xerr=x0, x_start=x_start)
    end

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end