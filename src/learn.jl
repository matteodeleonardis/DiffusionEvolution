function log_likelihood(x::Pars,  data::Data, λ::Float64)

    ll = 0.0
    
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(x, t, data, data.d)
        ll += log(det(Σ + λ*I(data.d))) + weighted_batch_dot(data.round[t].w, (data.round[t].x .- μ), inv(Σ + λ*I(data.d)))
    end

    return ll/length(data.round)
end


function optim_wrapper(x::Pars, g::Pars, data::Data, λ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data, λ)
    end

    g .= gs[1]
    return ll
end



function learn_nlopt(data::Data; x0=randn(npars(data.d)), 
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, λ=0.0)

    opt = Opt(alg, npars(data.d))
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data, λ)
    (minf, minx, status) = optimize(opt, x0)
    return (minf, minx, status)
end


function learn_gd(data::Data; x0=randn(npars(data.d)), epochs=(1,), η=(0.001,), λ=0.0, verbose=false)

    @assert length(epochs)==length(η)
    vals = zeros(sum(epochs))
    g = zeros(npars(data.d))
    for k in eachindex(epochs)
        last_epochs = k > 1 ? epochs[k-1] : 0
        for it in 1:epochs[k]
            x0 .-= η[k] * g
            vals[last_epochs+it] = optim_wrapper(x0, g, data, λ)
            if verbose
                println("ieter $(last_epochs+it)/$(sum(epochs)): ll=$(vals[last_epochs+it])")
            end
        end
    end

    return vals, x0
end




