function log_likelihood(x::Pars,  data::Data, γ::Float64, λ::Float64, prior_x::Float64, prior_γ::Float64, ϵ::Float64)

    ll = 0.0
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(x, γ, data.time[t], data.x0, data.d, ϵ, λ)
        ll += logdet(Σ) + weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), inv(Σ))
        ll += data.d*log2pi
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


function optim_wrapper(x::Pars, g::Pars, data::Data, γ::Float64, λ::Float64, prior_x::Float64, prior_γ::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data, γ, λ, prior_x, prior_γ, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_nlopt(data::Data; x0=randn(npars(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_x=0.0, γ=1.0, epsilon=0.0)

    opt = Opt(alg, npars(data.d))
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data, γ, lambda, prior_x, 0.0, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=false)
    end

    x_start = copy(x0)

    (minf, minx, status) = NLopt.optimize!(opt, x0)

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end


function optim_wrapper_only_gamma(x::Pars, g::Pars, data::Data, γ::Float64, λ::Float64, prior_γ::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(γ) do par
        ll = log_likelihood(x, data, par, λ, 0.0, prior_γ, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_nlopt_only_gamma(data::Data; x0, γ0=1.0, alg=:LD_LBFGS, xtol_rel=0.0, 
    ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_gamma=0.0, epsilon=0.0)

    opt = Opt(alg, 1)
    lb = [1e-12]
    opt.lower_bounds = lb
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (γ,g) -> optim_wrapper_only_gamma(x0, g, data, γ[1], lambda, prior_gamma, epsilon)

    γ_start = γ0

    (minf, minγ, status) = NLopt.optimize!(opt, [γ0])

    return (minf=minf, minγ=minγ, status=status, nevals=opt.numevals, γ_start=γ_start)
end


