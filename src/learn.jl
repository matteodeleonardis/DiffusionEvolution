function log_likelihood(x::Pars,  data::Data, λ::Float64, prior::Float64)

    ll = 0.0
    T = length(data.round)
    for t in 1:T-1
        μ, Σ = compute_parameters(x, t, data, data.d)
        ll += log(det((1.0-λ)*Σ + λ*I(data.d))) + weighted_batch_dot(data.round[t+1].w, (data.round[t+1].x .- μ), inv((1.0-λ)Σ + λ*I(data.d)))
    end

    ll /= length(data.round)
    if prior > 0.0
        for i in eachindex(x)
            ll += prior*(x[i]^2)
        end
    end

    return  ll
end


function optim_wrapper(x::Pars, g::Pars, data::Data, λ::Float64, prior::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data, λ, prior)
    end

    g .= gs[1]
    return ll
end



function learn_nlopt(data::Data; x0=randn(npars(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, λ=0.0,
    prior=0.0)

    opt = Opt(alg, npars(data.d))
    lb = fill(-Inf, npars(data.d))
    #lb[gamma_index(data.d)] = 1e-12
    opt.lower_bounds = lb
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data, λ, prior)

    if initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    (minf, minx, status) = NLopt.optimize(opt, x0)
    return (minf, minx, status)
end


function learn_gd(data::Data; x0=randn(npars(data.d)), initialize=-1,
    epochs=(1,), η=(0.001,), λ=0.0, prior=0.0, verbose=false, progress=false)

    @assert length(epochs)==length(η)
    vals = zeros(sum(epochs))

    if initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    g = zeros(npars(data.d))
    prog = Progress(sum(epochs))
    for k in eachindex(epochs)
        last_epochs = k > 1 ? epochs[k-1] : 0
        for it in 1:epochs[k]
            x0 .-= η[k] * g
            vals[last_epochs+it] = optim_wrapper(x0, g, data, λ, prior)
            if verbose
                println("iter $(last_epochs+it)/$(sum(epochs)): ll=$(vals[last_epochs+it])")
            end
            if progress
                next!(prog)
            end
        end
    end

    return vals, x0
end


function learn_optim(data; x0=randn(npars(data.d)), initialize=-1, λ=0.0, prior=0.0, algorithm=Optim.Adam())

    if initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    f(x) = log_likelihood(x, data, λ, prior)
    function g!(G, x) 
            gs = gradient(x) do par
            log_likelihood(par, data, λ, prior)
        end

        G .= gs[1]
    end

    return Optim.optimize(f, g!, x0, algorithm)
end




