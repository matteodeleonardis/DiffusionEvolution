module DiffusionEvolution

    using LinearAlgebra, Flux, NLopt, StatsBase, Optim, Distributions, FastaIO, BioSeqInt, MultivariateStats, PlmDCA, NPZ, Random, JLD2
    using PyPlot, PottsGauge, LogExpFunctions, DelimitedFiles

    import PyPlot.subplots
    subplots(x, y ,d) = PyPlot.subplots(x, y, figsize=(d*y, d*x))

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

    include("inference.jl")
    export infer_series, fit_series, get_params_tens

    include("contacts.jl")
    export compute_norm, corr_APC, compute_true_positives, compute_frob_norm, contact_plot

    include("analysis.jl")
    export compute_scores, print_contact_plot, compute_ppv, plot_distribution

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
        maxtime=-1, maxeval=-1,

        # optimization options for Optim
        optim_alg=Optim.LBFGS(), 
        
        #tolerance for stopping criteria
        stop_tol...)

        @assert opt_pkg in [:NLopt, :Optim] "Only :NLopt and :Optim are allowed for [opt_pkg] option."

        data_maxoutdim, pca = process_data(file_nat, file_wt, file_rounds, times; weight=weight, maxoutdim=maxoutdim)
        data = subdata(data_maxoutdim, d)

        x0 = randn(npars_gamma(data.d))

        if opt_pkg == :NLopt
            results =learn_gamma_nlopt(data; x0=x0, initialize=initialize, 
                lambda=lambda, prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma,
                alg=nlopt_alg, maxtime=maxtime, maxeval=maxeval, stop_tol...)
        elseif opt_pkg == :Optim
            results = learn_gamma_optim(data; x0=x0, initialize=initialize, 
                lambda=lambda, prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma, 
                alg=optim_alg, stop_tol...)
        end

        return data, pca, results
    end

    export learn

    include("analysis/train_and_analysis_dhfr.jl")
    export run_analysis_dhfr

    include("analysis/train_and_analysis_pse1.jl")
    export run_analysis_pse1
end
