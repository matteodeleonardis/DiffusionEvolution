function log_likelihood(x::Pars,  data::Data, γ::Float64, λ::Float64, prior_J::Float64, prior_theta::Float64, prior_γ::Float64, ϵ::Float64)

    ll = 0.0
    for t in eachindex(data.round)
        μ, Σ = compute_parameters(x, γ, data.time[t], data.x0, data.d, ϵ, λ)
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


function optim_wrapper(x::Pars, g::Pars, data::Data, γ::Float64, λ::Float64, prior_J::Float64, prior_theta::Float64, ϵ::Float64)

    if length(g)==0
        g = zeros(length(x))
    end

    ll = 0.0
    gs = gradient(x) do par
        ll = log_likelihood(par, data, γ, λ, prior_J, prior_theta, 0.0, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_nlopt(data::Data; x0=randn(npars(data.d)), initialize=-1,
    alg=:LD_LBFGS, xtol_rel=0.0, ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_J=0.0, prior_theta=0.0, gamma=1.0, epsilon=0.0)

    opt = Opt(alg, npars(data.d))
    opt.xtol_rel=xtol_rel
    opt.ftol_rel=ftol_rel
    opt.xtol_abs=xtol_abs
    opt.ftol_abs=ftol_abs
    opt.maxeval=maxeval
    opt.maxtime=maxtime

    opt.min_objective = (x,g) -> optim_wrapper(x, g, data, gamma, lambda, prior_J, prior_theta, epsilon)

    if initialize == 0
        init_id!(x0, d=data.d)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
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
        ll = log_likelihood(x, data, par, λ, 0.0, 0.0, prior_γ, ϵ)
    end

    g .= gs[1]
    return ll
end


function learn_nlopt_only_gamma(data::Data; x0, gamma0=1.0, alg=:LD_LBFGS, xtol_rel=0.0, 
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

    γ_start = gamma0

    (minf, minγ, status) = NLopt.optimize!(opt, [gamma0])

    return (minf=minf, min_gamma=minγ, status=status, nevals=opt.numevals, gamma_start=γ_start)
end


function iterative_maximization(data::Data; x=randn(npars(data.d)), gamma=1.0, initialize=-1, alg=:LD_LBFGS, xtol_rel=0.0, 
    ftol_rel=0.0, xtol_abs=0.0, ftol_abs=0.0, maxtime=-1, maxeval=-1, lambda=0.0,
    prior_J=0.0, prior_theta=0.0, prior_gamma=0.0, epsilon=0.0, iterations=1, verbose=true, logfile::String)

    file_log = open(logfile, "w")

    ll_iter = zeros(iterations+1)

    #setting initial condition for x
    if initialize == 0
        init_id!(x, d=data.d)
    elseif initialize>0
        init_cov!(x, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    ll_iter[1] = log_likelihood(x,data, gamma, lambda, prior_J, prior_theta, prior_gamma, epsilon)

    γ_vec = [gamma]

    for iter in 1:iterations

        #parameter optimization
        opt_x = Opt(alg, npars(data.d))
        opt_x.xtol_rel=xtol_rel
        opt_x.ftol_rel=ftol_rel
        opt_x.xtol_abs=xtol_abs
        opt_x.ftol_abs=ftol_abs
        opt_x.maxeval=maxeval
        opt_x.maxtime=maxtime

        opt_x.min_objective = (x_fx,g_fx) -> optim_wrapper(x_fx, g_fx, data, gamma, lambda, prior_J, prior_theta, epsilon)
        x_start_x = copy(x)
        f_start_x = log_likelihood(x, data, gamma, lambda, prior_J, prior_theta, 0.0, epsilon)
        (minf_x, minx, status_x) = NLopt.optimize!(opt_x, x)

        if verbose
            Δx = maximum(abs.(x_start_x .- x))
            Δfx = abs(f_start_x - minf_x)
            println(file_log, "Parameters optimization iteration $iter exited with status $status_x. |Δx|=$(Δx), |Δfx|=$(Δfx)")
            flush(stdout)
        end

        #gamma optimization
        opt_γ = Opt(alg, 1)
        lb_γ = [1e-12]
        opt_γ.lower_bounds = lb_γ
        opt_γ.xtol_rel=xtol_rel
        opt_γ.ftol_rel=ftol_rel
        opt_γ.xtol_abs=xtol_abs
        opt_γ.ftol_abs=ftol_abs
        opt_γ.maxeval=maxeval
        opt_γ.maxtime=maxtime

        opt_γ.min_objective = (γ_fγ,g_fγ) -> optim_wrapper_only_gamma(x, g_fγ, data, γ_fγ[1], lambda, prior_gamma, epsilon)
        γ_start = γ_vec[1]
        f_start_γ = log_likelihood(x, data, γ_vec[1], lambda, 0.0, 0.0, prior_gamma, epsilon)
        (minf_γ, minγ, status_γ) = NLopt.optimize!(opt_γ, γ_vec)

        if verbose
            Δγ = abs(γ_start - γ_vec[1])
            Δfγ = abs(f_start_γ - minf_γ)
            println(file_log, "γ optimization iteration $iter exited with status $status_γ. |Δγ|=$(Δγ), |Δfγ|=$(Δfγ) \n")
            flush(stdout)
        end

        #end iteration
        ll_iter[iter+1] = log_likelihood(x, data, γ_vec[1], lambda, prior_J, prior_theta, prior_gamma, epsilon)
    end
    close(file_log)

    return (minx=x, min_gamma=γ_vec[1], ll_iter=ll_iter)
end


function optimize_pars_gd!(data::Data; x=randn(npars(data.d)), gamma=1.0, initialize=-1, 
    lambda=0.0, prior_J=0.0, prior_theta=0.0, epsilon=0.0, 
    eta=0.001, iterations=1)

    ll_iter = zeros(iterations+1)

    #setting initial condition for x
    if initialize == 0
        init_id!(x, d=data.d)
    elseif initialize>0
        init_cov!(x, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    ll_iter[1] = log_likelihood(x, data, gamma, lambda, prior_J, prior_theta, 0.0, epsilon)
    g_x = zeros(n_pars(data.d))
    for it in 1:iterations
        optim_wrapper(x, g_x, data, gamma, lambda, prior_J, prior_theta, epsilon)
        x .-= (eta * g_x)
        ll_iter[it+1] = log_likelihood(x, data, gamma, lambda, prior_J, prior_theta, 0.0, epsilon)
    end
end


function learn_optim(data::Data; x0=randn(npars(data.d)), initialize=-1, gamma,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, epsilon=0.0, g_tol=1e-8, f_tol=0.0, x_tol=0.0)

    if initialize == 0
        init_id!(x0, d=data.d)
    elseif initialize>0
        init_cov!(x0, data.round[initialize].x, data.round[initialize].w, d=data.d)
    end

    function fg!(F,G,x)

        ll = 0.0
        if G !== nothing
            ll = optim_wrapper(x, G, data, gamma, lambda, prior_J, prior_theta, epsilon)
        elseif F!== nothing
            ll = log_likelihood(x, data, gamma, lambda, prior_J, prior_theta, 0.0, epsilon)
        end

        return ll
    end

    res = Optim.optimize(Optim.only_fg!(fg!), x0, alg, Optim.Options(g_tol=g_tol, f_tol=f_tol, x_tol=x_tol))

    return res
end



function line_search_optimization(data::Data; x0=randn(npars(data.d)), initialize=-1, 
    n_points, iterations=1, gamma_upper=1.0, gamma_lower=-1.0,
    alg=Optim.LBFGS(), lambda=0.0, prior_J=0.0, prior_theta=0.0, epsilon=0.0, g_tol=1e-8, f_tol=0.0, x_tol=0.0)

    opt_results = Vector{Any}(undef, iterations)
    opt_gamma = Vector{Any}(undef, iterations)

    if gamma_lower < 0.0
        gamma_lower = gamma_upper/n_points
    end
    opt_linesearch = Vector{Any}(undef, n_points)
    

    for it in 1:iterations
        println("Iteration $it")
        println("gamma bounds: ($gamma_lower, $gamma_upper)")
        tau_upper = inv(gamma_lower)
        tau_lower = inv(gamma_upper)
        tau_range = LinRange(tau_lower, tau_upper, n_points)
        gamma_values= inv.(reverse(tau_range))
        println(gamma_values)
        Threads.@threads for i in eachindex(gamma_values)
            opt_linesearch[i] = learn_optim(data, x0=x0, initialize=initialize, gamma=gamma_values[i], alg=alg, lambda=lambda, prior_J=prior_J,
                prior_theta=prior_theta, epsilon=epsilon, g_tol=g_tol, f_tol=f_tol, x_tol=x_tol)
        end
        min_i = argmin(map(x->x.minimum, opt_linesearch))
        if min_i==1
            println("Optimal gamma hit the lower border, something bad happened.") 
            break
        elseif min_i==length(gamma_values)
            println("Optimal gamma hit the upper border, something bad happened.") 
            break
        end
        opt_results[it] = opt_linesearch[min_i]
        opt_gamma[it] = gamma_values[min_i]
        println("optimal gamma: $(gamma_values[min_i])")
        for i in min_i-1:-1:1
            if opt_linesearch[i].minimum > opt_linesearch[min_i].minimum
                gamma_lower = gamma_values[i]
                break
            end
        end
        for i in min_i+1:length(opt_linesearch)
            if opt_linesearch[i].minimum > opt_linesearch[min_i].minimum
                gamma_upper = gamma_values[i]
                break
            end
        end
    end

    return opt_results, opt_gamma
end
    