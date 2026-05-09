import Pkg
Pkg.activate("..")

using MultivariateStats, JLD2, PyPlot, FastaIO, LaTeXStrings, StatsBase, LinearAlgebra
using DiffusionEvolution: process_data, get_gamma, compute_parameters, compute_J

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)

function collect_model_params_files(base_dir::AbstractString)
    target_name = "pse1_analysis_d_*.pars.jld2"

    # We mimic awk's regex capture of the d value from the full path
    re = r"/pse1_analysis_d_([0-9]+)/pse1_analysis_d_[0-9]+\.pars\.jld2$"

    pairs = Tuple{Int, String}[]  # (d, path)

    for (root, _dirs, files) in walkdir(base_dir)
        for f in files
            # quick name filter like find -name ...
            occursin(r"^pse1_analysis_d_\d+\.pars\.jld2$", f) || continue

            full = joinpath(root, f)

            m = match(re, full)
            m === nothing && continue

            d = parse(Int, m.captures[1])
            push!(pairs, (d, full))
        end
    end

    sort!(pairs, by = first)           # sort -n -k1,1
    return [p[2] for p in pairs]        # cut -f2-
end

function main()
    file_model_params = collect_model_params_files("/home/students/s301803/CODE/DiffusionEvolution/results/pse1/run0")
    d_labels = [parse(Int, split(split(f, "/")[10], "_")[4]) for f in file_model_params]

    #load data
    data_dir = joinpath(@__DIR__, "../../../data")

    file0 = joinpath(data_dir, "pse1/PSE1.fas")
    file1 = joinpath(data_dir, "pse1/Rnd10.fas")
    file2 = joinpath(data_dir, "pse1/Rnd20_init.fas")

    times = [10, 20]

    # natural sequences
    file_nat = joinpath(data_dir, "pse1/PSE1_clean.fasta")
    file_rounds = [file1, file2]

    A = 21
    L = length(readfasta(file0)[1][2])

    data, pca = process_data(file_nat, file0, file_rounds, times; whiten=false, weight=false,
                                    d=d_labels[end], extreme=false)

    epsilon_J = 1.0e-9
    epsilon_sigma = 1.0e-12
    lambda = 0.01

    cov_corr_equilibrium = zeros(length(d_labels))
    cov_diff_equilibrium = zeros(length(d_labels))
    cov_corr_last = zeros(length(d_labels))
    cov_diff_last = zeros(length(d_labels))

    fig_scatter, ax_scatter = subplots(2, length(d_labels), 6)

    for i in eachindex(file_model_params)
        file = file_model_params[i]
        d = d_labels[i]
        x_pars = JLD2.load(file)["x_opt"]

        J = compute_J(x_pars, d, epsilon_J)
        C = cholesky(J)
        sigma = C \ I(d)
        sigma = 0.5 * (sigma + sigma')
        sigma += epsilon_sigma*I(d)

        gamma = get_gamma(x_pars, d)
        _, model_cov = compute_parameters(x_pars, gamma, times[end], data.x0[1:d], d, epsilon_J, lambda)

        empirical_cov = cov(data.round[end].x[1:d, :], Weights(data.round[end].w), 2)

        cov_corr_equilibrium[i] = cor(vec(sigma), vec(empirical_cov))
        cov_diff_equilibrium[i] = sum(abs2, sigma .- empirical_cov)/abs2(d)
        cov_corr_last[i] = cor(vec(model_cov), vec(empirical_cov))
        cov_diff_last[i] = sum(abs2, model_cov .- empirical_cov)/abs2(d)

        ax_scatter[1,i].scatter(vec(sigma), vec(empirical_cov), s=4)
        ax_scatter[1,i].set_xlabel("equilibrium model sigma")
        ax_scatter[1,i].set_ylabel("empirical covariance")
        ax_scatter[1,i].set_title("d=$d")

        ax_scatter[2,i].scatter(vec(model_cov), vec(empirical_cov), s=4)
        ax_scatter[2,i].set_xlabel("model covariance")
        ax_scatter[2,i].set_ylabel("empirical covariance")
        ax_scatter[2,i].set_title("d=$d")
    end

    fig_scatter.savefig("pse1_cov_empirical_vs_model.scatter.png", format="png", bbox_inches="tight")

    fig_cov, ax_cov = subplots(1, 2, 6)
    ax_cov[1].plot(d_labels, cov_corr_equilibrium, marker="o", label="equilibrium")
    ax_cov[1].plot(d_labels, cov_corr_last, marker="o", label="last round")
    ax_cov[1].legend()
    ax_cov[1].set_xlabel("d")
    ax_cov[1].set_ylabel("pearson")

    ax_cov[2].plot(d_labels, cov_diff_equilibrium, marker="o", label="equilibrium")
    ax_cov[2].plot(d_labels, cov_diff_last, marker="o", label="last round")
    ax_cov[2].legend()
    ax_cov[2].set_yscale(:log)
    ax_cov[2].set_xlabel("d")
    ax_cov[2].set_ylabel("average squadred difference")

    #ax_cov.set_title("Model vs empirical covariance at last round")
    fig_cov.savefig("pse1_cov_empirical_vs_model.png", format="png", bbox_inches="tight")
end

main()