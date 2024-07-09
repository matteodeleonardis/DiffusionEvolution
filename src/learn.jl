function log_likelihood(x::Pars,  data::Data)

    ll = 0.0
    
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(x, t, data, data.d)
        ll += log(det(Σ)) + weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), inv(Σ))
    end

    return ll
end


function optim_wrapper(x::Pars, g::Pars, data::Data)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data)
    end

    g .= gs[1]
    return ll
end



function learn_nlopt(data::Data; x0=randn(npars(data.d)), 
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1)

    opt = Opt(alg, npars(data.d))
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data)
    return optimize(opt, x0)
end


function learn_ga(data::Data; x0=randn(npars(data.d)), epochs=(1,), η=(0.001,))

    @assert length(epochs)==length(η)
    vals = zeros(sum(epochs))
    g = zeros(npars(data.d))
    for k in eachindex(epochs)
        last_epochs = k > 1 ? epochs[k-1] : 0
        for it in 1:epochs[k]
            x0 .-= η[k] * g
            vals[last_epochs+it] = optim_wrapper(x0, g, data)
        end
    end

    return vals, x0
end




