function log_likelihood_gamma(x::Pars,  data::Data, λ::Float64, prior::Float64, ϵ::Float64)

    ll = 0.0
    T = length(data.round)
    for t in 1:T-1
        μ, Σ = compute_parameters(x, data.time[t+1], data.x0, data.d)
        ll += logdet((1.0-λ)*Σ + λ*I(data.d)) + weighted_batch_dot(data.round[t+1].w, (data.round[t+1].x .- μ), inv((1.0-λ)*Σ + λ*I(data.d)))
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


function learn_gamma_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), λ=0.0, prior=0.0, rescale=false, epsilon=0.0, x_abstol=0.0, x_reltol=0.0, 
    f_abstol=0.0, f_reltol=0.0, g_abstol=1e-8, err_file="err_file")

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true, 
            rescale=rescale)
    end

    lower = vcat(fill(-Inf, npars(data.d)), 1e-12)
    upper = fill(+Inf, npars_gamma(data.d))

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = try optim_wrapper_gamma(x, G, data, λ, prior, epsilon)
            catch e
                println(e)
                @save err_file*".jld2" x
            end

        elseif F!== nothing
            ll = try log_likelihood_gamma(x, data, λ, prior, epsilon)
            catch e
                println(e)
                @save err_file*".jld2" x
            end
        end

        return ll
    end

    res = Optim.optimize(Optim.only_fg!(fg!), lower, upper, x0, Fminbox(alg), 
        Optim.Options(x_abstol=x_abstol, x_reltol=x_reltol, f_abstol=f_abstol, f_reltol=f_reltol,
        g_abstol=g_abstol))

    return res
end


function learn_gamma_unconstrained_optim(data::Data; x0=randn(npars_gamma(data.d)), initialize=-1,
    alg=Optim.LBFGS(), λ=0.0, prior=0.0, rescale=false, epsilon=0.0)

    if initialize == 0
        init_id!(x0, d=data.d, init_gamma=true)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d, init_gamma=true, 
            rescale=rescale)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = optim_wrapper_gamma(x, G, data, λ, prior, epsilon)
        elseif F!== nothing
            ll = log_likelihood_gamma(x, data, λ, prior, epsilon)
        end

        return ll
    end

    res = Optim.optimize(Optim.only_fg!(fg!), x0, alg, 
        Optim.Options(x_abstol=x_abstol, x_reltol=x_reltol, f_abstol=f_abstol, f_reltol=f_reltol,
        g_abstol=g_abstol))

    return res
end