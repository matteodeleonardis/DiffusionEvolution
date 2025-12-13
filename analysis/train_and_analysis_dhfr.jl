using DiffusionEvolution, PyPlot, Distributions, Random

import subplots
subplots(x, y ,d; kargs...) = PyPlot.subplots(x, y, figsize=(d*y, d*x), kargs...)

function run_training(;d, opt_pkg, output_root)

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

    data, pca, results = learn(file_nat, file0, [file1, file2, file3, file4, file5, file15], times;
        weight=false, maxoutdim=d, opt_pkg=opt_pkg, d=d, initialize=length(times), lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma)


    x_opt = results.minimizer
    #save parameters
    @save output_root * ".pars.jld2" x_opt

    #save data
    @save output_root * ".data.jld2" data
    @save output_root * ".pca.jld2" pca

    #plot data
    ax_emp_dist, fig_emp_dist = subplots(1, length(times), 6, squeeze=true)
    for i in eachindex(times)
        ax_emp_dist[1,i].hist2d(data.x[i][:,1], data.x[i][:,2], bins=50)
        ax_emp_dist[1,i].plot([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
        ax_emp_dist[1,i].set_title("t=$(times[i])")
        ax_emp_dist[1,i].set_xlabel("PC1")
        ax_emp_dist[1,i].set_ylabel("PC2")
    end
    fig_emp_dist.savefig(output_root * ".emp_dist.png", fmt="png", bbox_inches="tight")

 

    #plot inferred distribution
    mu, sigma, eq_theta, eq_sigma = infer_series(x_opt, data, times; lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma)
    ax_inf_dist, fig_inf_dist = subplots(1, length(times) + 1, 6, squeeze=false)
    for i in eachindex(times)
        samples = rand(MultivariateNormal(mu[times[i]], sigma[times[i]]), 1000)
        ax_inf_dist[1,i].hist2d(samples[:,1], samples[:,2], bins=50)
        ax_inf_dist[1,i].plot([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
        ax_inf_dist[1,i].set_title("t=$(times[i])")
        ax_inf_dist[1,i].set_xlabel("PC1")
        ax_inf_dist[1,i].set_ylabel("PC2")
    end
    fig_inf_dist.savefig(output_root * ".inf_dist.png", fmt="png", bbox_inches="tight")

end

run_training(d=1, opt_pkg=Optim, output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test")
