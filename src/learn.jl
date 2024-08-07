function safe_log(x::Float64, ϵ::Float64)

    return log(x + ϵ)
end

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


function optim_wrapper_gamma_track(x::Pars, g::Pars, data::Data, λ::Float64, prior::Float64, 
    history::MVHistory, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood_gamma(par, data, λ, prior, ϵ)
    end

    push!(history, :x, x)
    push!(history, :log_likelihood, ll)

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
    catch
        return (x_err = x0, nevals=opt.numevals)
    end

    return (minf, minx, status, opt.numevals)
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
        println(e)
        return (xerr=x0, x_start=x_start)
    end

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start)
end


function learn_gamma_nlopt_track(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, λ=0.0,
    prior=0.0, epsilon=0.0)

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

    history = MVHistory()

    opt.min_objective = (x,g) -> optim_wrapper_gamma_track(x, g, data, λ, prior, history, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true)
    end

    x_start = copy(x0)

    (minf, minx, status) = try NLopt.optimize!(opt, x0)
    catch e
        println("exception")
        if :msg in fieldnames(typeof(e))
            println(e.msg)
        end

        return (x=x0, nevals=opt.numevals, x_start=x_start, history=history)
    end

    return (minf=minf, minx=minx, status=status, nevals=opt.numevals, x_start=x_start, history=history)
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
            vals[last_epochs+it] = optim_wrapper(x0, g, data, 1.0, λ, prior)
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


function learn_gamma_gd(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
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
            vals[last_epochs+it] = optim_wrapper_gamma(x0, g, data, λ, prior)
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

    f(x) = log_likelihood(x, data, 1.0, λ, prior)
    function g!(G, x) 
            gs = gradient(x) do par
            log_likelihood(par, data, 1.0, λ, prior)
        end

        G .= gs[1]
    end

    return Optim.optimize(f, g!, x0, algorithm)
end




