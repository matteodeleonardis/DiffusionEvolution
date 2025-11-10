module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim, Distributions, JLD2, FastaIO, BioSeqInt, MultivariateStats

    const Pars = Vector{Float64}
    const log2pi = log(2.0*π)

    include("data.jl")
    export collect_data, Data, data_entropy

    include("parameters.jl")
    export which_par

    include("learn.jl")
    export learn_nlopt, iterative_maximization, optimize_pars_gd!, learn_optim, line_search_optimization

    include("learn_gamma.jl")
    export learn_gamma_nlopt, learn_gamma_optim, learn_gamma_unconstrained_optim, optimize_gd!

    include("learn_gamma_l1.jl")
    export learn_gamma_nlopt_l1, learn_gamma_optim_l1, optimize_gd_l1!

    include("learn_small_gamma.jl")
    export learn_small_gamma_nlopt, optimize_small_gamma_gd!, learn_small_gamma_optim

    include("utils.jl")
    export compute_energy, compute_weight, get_potts_params, compute_entropy

    #simulation
    include("simulate/simulate_ou.jl")
    export random_pars, simulate_ou_process

    function learn(file_nat, file_wt, file_rounds, times; 

        # options for data processing
        weight=false, maxoutdim=10,

        # generic options
        opt_pkg::Symbol, initialize=-1,  
        lambda=0.01, prior_J=0.0001, prior_theta=0.0001, prior_gamma=0.0001, epsilon_J=1.0e-9, epsilon_sigma=1.0e-12, d=1,

        # optimization options for NLopt
        nlopt_alg=:LD_LBFGS, 
        xtol_rel=1.0e-7, ftol_rel=1.0e-7, xtol_abs=1.0e-7, ftol_abs=1.0e-7, 
        maxtime=-1, maxeval=-1,

        # optimization options for Optim
        optim_alg=Optim.LBFGS(), g_tol=1.0e-4, f_tol=1.0e-7, x_tol=1.0e-5)

        @assert opt_pkg in [:NLopt, :Optim] "Only :NLopt and :Optim are allowed for [opt_pkg] option."

        data_maxoutdim = process_data(file_nat, file_wt, file_rounds, times; weight=weight, maxoutdim=maxoutdim)
        data = subdata(data_maxoutdim, d)

        x0 = randn(npars_gamma(data.d))

        if opt_pkg == :NLopt
            results =learn_gamma_nlopt(data; x0=x0, initialize=initialize, 
                lambda=lambda, prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma,
                alg=nlopt_alg, xtol_rel=xtol_rel, ftol_rel=ftol_rel, xtol_abs=xtol_abs, ftol_abs=ftol_abs, 
                maxtime=maxtime, maxeval=maxeval)
        elseif opt_pkg == :Optim
            results = learn_gamma_optim(data; x0=x0, initialize=initialize, 
                lambda=lambda, prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma, 
                alg=optim_alg, g_tol=g_tol, f_tol=f_tol, x_tol=x_tol)
        end

        return data, results
    end

    export learn
end
