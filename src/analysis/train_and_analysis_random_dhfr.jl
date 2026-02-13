function run_analysis_random_dhfr(;d, opt_pkg, output_root, contacts_file)
    # d=2
    # opt_pkg=:Optim
    # output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test"
    # contacts_file="/home/matteo/Projects/mDHFR/contacts_prediction/contact_map.npy"

    data_dir = joinpath(@__DIR__, "../../data")

    # experiment data
    file0 = joinpath(data_dir, "dhfr/mDHFR.fasta")
    file1 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_1.fasta")
    file2 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_2.fasta")
    file3 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_3.fasta")
    file4 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_4.fasta")
    file5 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_5.fasta")
    file15 = joinpath(data_dir, "random_dhfr/random_dhfr_rnd_15.fasta")

    times = [1,2,3,4,5, 15]


    # natural sequences
    file_nat = joinpath(data_dir, "dhfr/mDHFR_clean.fasta")
    #file_nat = file1

    #training
    lambda = 0.01
    epsilon_J = 1e-9
    epsilon_sigma = 1e-12
    prior_J = 0.01
    prior_theta = 0.01
    prior_gamma = 0.01 #prior_J * d^2
    whiten = false
    extreme = false

    #stopping criteria
    stop_tol = (g_abstol = 1.0e-4, x_abstol = 1.0e-5, x_reltol = 1.0e-5)

    results, model_settings, gamma_min, data, pca = DiffusionEvolution.learn(file_nat, file0, [file1, file2, file3, file4, file5, file15], times;
        whiten=whiten, extreme=extreme, weight=false, opt_pkg=opt_pkg, d=d, initialize=length(times), 
        prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma,
        lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma, stop_tol...)


    x_opt = results.minimizer
    g_res = zeros(length(x_opt))
    ll_opt = DiffusionEvolution.optim_wrapper_gamma(x_opt, g_res, data, lambda, 
        prior_J, prior_theta, prior_gamma, 
        epsilon_J, epsilon_sigma)
    #if gamma is at the border, set gradient to zero
    if x_opt[end] <= 1e-12
        g_res[end] = 0.0
    end
    J_opt = DiffusionEvolution.compute_J(x_opt, d, epsilon_J)
    fig_J = figure()
    ax_J = gca()
    img_J = ax_J.imshow(J_opt)
    colorbar(img_J, ax=ax_J)
    fig_J.savefig(output_root * ".J_inferred.png", format="png", bbox_inches="tight")
    open(output_root * ".optimization.log", "w") do io
        print(io, "*** Optimization Results *** \n ", results, "\n")
        theta_opt = DiffusionEvolution.compute_theta(x_opt, d)
        print(io, "Gamma: ", DiffusionEvolution.get_gamma(results.minimizer, d), "\n")
        print(io, "Gamma_min_empirical: ", gamma_min, "\n")
        print(io, "extrema |J|: ", extrema(abs.(J_opt)), "\n")
        print(io, "extrema |h|: ", extrema(abs.(theta_opt)), "\n")
        print(io, "x[gamma]: ", results.minimizer[end], "\n")
        print(io, "extrema |x[J]|: ", extrema(abs.(results.minimizer[1:DiffusionEvolution.n_couplings(d)])), "\n")
        print(io, "extrema |x[h]|: ", extrema(abs.(results.minimizer[DiffusionEvolution.n_couplings(d)+1:end-1])), "\n")
        print(io, "max |g|: ", maximum(abs.(g_res)), "\n")
        print(io, "max |g|/|f|: ", maximum(abs.(g_res))/ll_opt, "\n")
        print(io, "\n")
        print(io, "MODEL SETTINGS: \n ")
        for (name, value) in pairs(model_settings)
            println(io, "$(name): $(value)")
        end

    end

    ratio_tvar = pca.tprinvar / pca.tvar

    #save parameters and settings
    @save output_root * ".pars.jld2" x_opt gamma_min ratio_tvar
    @save output_root * ".settings.jld2" model_settings

    #plot inferred distribution
    plot_distribution(x_opt, data, times, output_root; lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma)

    #parameters as tensors
    fasta_wt = readfasta(file0)
    wt = aa2int.(uppercase(fasta_wt[1][2]))
    L = length(fasta_wt[1][2])
    A = 21
    J_tens, h_tens, gamma = get_params_tens(x_opt, pca, d, epsilon_J, A, L; whiten=whiten, eps_warning=1.0e-4, set_zero=false)

    #computing scores
    frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc = compute_scores(J_tens, h_tens, wt, L, output_root)

    true_contacts=npzread(contacts_file)

    #ppv
    compute_ppv(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)

    #contact plot
    print_contact_plot(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)
end

# d = parse(Int, ARGS[1])
# output_root = ARGS[2]

# run_training(d=d, opt_pkg=:Optim, output_root=output_root,
#     contacts_file="/home/students/s301803/dhfr_neutral_evolution/DHFR/contact_map.npy")
