function log_likelihood(x::Pars,  data::Data, γ::Float64, λ::Float64, prior::Float64, ϵ::Float64)

    ll = 0.0
    T = length(data.round)
    for t in 1:T-1
        μ, Σ = compute_parameters(x, γ, data.time[t+1], data.x0, data.d)
        ll += safe_log(det((1.0-λ)*Σ + λ*I(data.d)), ϵ) + weighted_batch_dot(data.round[t+1].w, (data.round[t+1].x .- μ), inv((1.0-λ)*Σ + λ*I(data.d)))
    end

    ll /= length(data.round)
    if prior > 0.0
        for i in eachindex(x)
            ll += prior*(x[i]^2)
        end
    end

    return  ll
end


function optim_wrapper(x::Pars, g::Pars, data::Data, γ::Float64, λ::Float64, prior::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data, γ, λ, prior, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_nlopt(data::Data; x0=randn(npars(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, λ=0.0,
    prior=0.0, γ=1.0, rescale=false, epsilon=0.0)

    opt = Opt(alg, npars(data.d))
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data, γ, λ, prior, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, rescale=rescale)
    end

    (minf, minx, status) = try NLopt.optimize!(opt, x0)
    catch e
        println("Optimization failed. \n", e)
        return (x_err = x0, nevals=opt.numevals)
    end

    return (minf, minx, status, opt.numevals)
end


