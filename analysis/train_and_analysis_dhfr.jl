import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise

using DiffusionEvolution,PyPlot, Distributions, Random, JLD2, Optim, NLopt

import PyPlot.subplots
subplots(x, y ,d) = PyPlot.subplots(x, y, figsize=(d*y, d*x))

function run_training(;d, opt_pkg, output_root)
    open(output_root * ".log", "w") do io
        redirect_stderr(io)
        redirect_stdout(io)
        # d=2
        # opt_pkg=:Optim
        # output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test"

        # experiment data
        file0 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/uniprot/mDHFR.fasta"
        file1 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/GenEarly/Oct10_QComp/Round1_Q15_C10_aa.aln"
        file2 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/GenEarly/Oct10_QComp/Round2_Q15_C10_aa.aln"
        file3 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/GenEarly/Oct10_QComp/Round3_Q15_C10_aa.aln"
        file4 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/GenEarly/Oct10_QComp/Round4_Q15_C10_aa.aln"
        file5 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/GenEarly/Oct10_QComp/Round5_Q15_C10_aa.aln"
        file15 = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/Gen15/Gen15_aa.aln"

        times = [1,2,3,4,5, 15]
    

        # natural sequences
        file_nat = "/home/matteo/Projects/mDHFR/dhfr_neutral_evolution/DHFR/uniprot/mDHFR_clean.fasta"

        #training
        lambda = 0.01
        epsilon_J = 1e-9
        epsilon_sigma = 1e-12

        data, pca, results = DiffusionEvolution.learn(file_nat, file0, [file1, file2, file3, file4, file5, file15], times;
            weight=false, maxoutdim=d, opt_pkg=opt_pkg, d=d, initialize=length(times), lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma)


        x_opt = results.minimizer
        print("*** Optimization Results *** \n ", results)
        #save parameters
        @save output_root * ".pars.jld2" x_opt

        #save data
        @save output_root * ".data.jld2" data
        @save output_root * ".pca.jld2" pca

        #plot data
        fig_emp_dist, ax_emp_dist = subplots(1, length(times), 6)
        for i in eachindex(times)
            ax_emp_dist[i].hist2d(data.round[i].x[1,:], data.round[i].x[2,:], weights=data.round[i].w, bins=50)
            ax_emp_dist[i].scatter([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
            ax_emp_dist[i].set_title("t=$(times[i])")
            ax_emp_dist[i].set_xlabel("PC1")
            ax_emp_dist[i].set_ylabel("PC2")
        end
        fig_emp_dist.savefig(output_root * ".emp_dist.png", format="png", bbox_inches="tight")

    

        #plot inferred distribution
        mu, sigma, eq_theta, eq_sigma = infer_series(x_opt, data, times; lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma)
        fig_inf_dist, ax_inf_dist = subplots(1, length(times), 6)
        for i in eachindex(times)
            samples = rand(MultivariateNormal(mu[times[i]], sigma[times[i]]), 1000)
            ax_inf_dist[i].hist2d(samples[1,:], samples[2,:], bins=50)
            ax_inf_dist[i].scatter([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
            ax_inf_dist[i].set_title("t=$(times[i])")
            ax_inf_dist[i].set_xlabel("PC1")
            ax_inf_dist[i].set_ylabel("PC2")
        end
        fig_inf_dist.savefig(output_root * ".inf_dist.png", format="png", bbox_inches="tight")
    
    end
end

run_training(d=2, opt_pkg=:Optim, output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test")
