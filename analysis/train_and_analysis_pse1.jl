import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution, PyPlot, Distributions, Random, JLD2, Optim, NLopt, PlmDCA, NPZ, FastaIO, PottsGauge, DelimitedFiles, BioSeqInt

import PyPlot.subplots
subplots(x, y ,d) = PyPlot.subplots(x, y, figsize=(d*y, d*x))

function run_training(;d, opt_pkg, output_root, contacts_file)
    # d=2
    # opt_pkg=:Optim
    # output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test"
    # contacts_file="/home/matteo/Projects/mDHFR/contacts_prediction/contact_map.npy"

    # experiment data
    file0 = "/home/students/s301803/diffusion_evolution/dev/data/pse1/PSE1.fas"
    file1 = "/home/students/s301803/diffusion_evolution/dev/data/pse1/Rnd10.fas"
    file2 = "/home/students/s301803/diffusion_evolution/dev/data/pse1/Rnd20_init.fas"

    times = [1,2,3,4,5, 15]


    # natural sequences
    file_nat = "/home/students/s301803/diffusion_evolution/dev/data/pse1/PSE1_clean.fasta"
    #file_nat = file1

    #training
    lambda = 0.01
    epsilon_J = 1e-9
    epsilon_sigma = 1e-12
    prior_J = 0.01
    prior_theta = 0.01
    prior_gamma = prior_J * d^2

    data, pca, results = DiffusionEvolution.learn(file_nat, file0, [file1, file2, file3, file4, file5, file15], times;
        weight=false, maxoutdim=d, opt_pkg=opt_pkg, d=d, initialize=length(times), 
        prior_J=prior_J, prior_theta=prior_theta, prior_gamma=prior_gamma, 
        lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma)


    x_opt = results.minimizer
    open(output_root * ".optimization.log", "w") do io
        print(io, "*** Optimization Results *** \n ", results, "\n")
        print(io, "Gamma: ", results.minimizer[end], "\n")
        print(io, "extrema |J|: ", extrema(abs.(results.minimizer[1:DiffusionEvolution.n_couplings(d)])), "\n")
        print(io, "extrema |h|: ", extrema(abs.(results.minimizer[DiffusionEvolution.n_couplings(d)+1:end-1])), "\n")
    end
    #save parameters
    @save output_root * ".pars.jld2" x_opt

    #save data
    @save output_root * ".data.jld2" data
    @save output_root * ".pca.jld2" pca


    #plot inferred distribution
    plot_distribution(x_opt, data, times, output_root; lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma)

    #parameters as tensors
    fasta_wt = readfasta(file0)
    wt = aa2int.(uppercase(fasta_wt[1][2]))
    L = length(fasta_wt[1][2])
    A = 21
    J_tens, h_tens, gamma = get_params_tens(x_opt, pca, d, epsilon_J, A, L, eps_warning=1.0e-4, set_zero=false)

    #computing scores
    frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc = compute_scores(J_tens, h_tens, wt, L, output_root)

    true_contacts=JLD2.load(contacts_file)["contacts"]

    #ppv
    compute_ppv(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)

    #contact plot
    print_contact_plot(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)
end

d = parse(Int, ARGS[1])

run_training(d=d, opt_pkg=:Optim, output_root="/home/students/s301803/diffusion_evolution/dev/analysis/pse1_analysis_d_$d/pse1_analysis_d_$d",
    contacts_file="/home/students/s301803/diffusion_evolution/dev/data/pse1/contact_map.jld2")
