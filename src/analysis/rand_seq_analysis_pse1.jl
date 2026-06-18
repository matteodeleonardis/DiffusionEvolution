function rand_seq_analysis_pse1(file_wt, file_rounds, file_nat, model_pars)

    times = [10, 20]

    dvals = [parse(Int, split(split(basename(p), ".")[1], "_")[end]) for p in model_pars]
    dmax = maximum(dvals)

    avg_mut_rate, d_hamm_init_data, weights_data = compute_average_mut_rate(file_rounds, file_wt)
    rand_samples, d_hamm_init_rand = produce_random_data(avg_mut_rate, file_wt, n_samples=10000)
    outname_d_hamm = joinpath(dirname(dirname(model_pars[1])), "d_hamm_init.png")
    fig_hamm, ax_hamm = subplots(1, length(times), 6)
    for t in eachindex(times)
        ax_hamm[t].hist(d_hamm_init_data[t], weights=weights_data[:,t], density=true, bins=collect(0:50), alpha=0.3, label="experimental data")
        ax_hamm[t].hist(d_hamm_init_rand[t], density=true, bins=collect(0:50), alpha=0.3, label="random data")
        ax_hamm[t].set_xlabel("Hamming distance")
        ax_hamm[t].set_ylabel("pdf")
        ax_hamm[t].legend()
        ax_hamm[t].set_title("Hamming distance from WT at t=$(times[t])")
    end
    fig_hamm.savefig(outname_d_hamm, format="png", bbox_inches="tight")
    println("Saved figure at $outname_d_hamm")
    close(fig_hamm)

    Z_random, w_random, Z_data, counts_data, wt_data, pca = collect_samples_data(file_rounds, file_wt, file_nat, rand_samples, dmax)

    x_1hot_wt = Float64.(reshape(Flux.onehotbatch(wt_data, collect(1:21)), :, 1))
    x_pca_wt = predict(pca, x_1hot_wt)
    
    x_1hot_random = Float64.(reshape(Flux.onehotbatch(Z_random, collect(1:21)), :, size(Z_random,2)))
    x_pca_random = predict(pca, x_1hot_random)

    x_1hot_data = Float64.(reshape(Flux.onehotbatch(Z_data, collect(1:21)), :, size(Z_data,2)))
    x_pca_data = predict(pca, x_1hot_data)

    p_values = zeros(length(model_pars), length(times))

    for i in eachindex(model_pars)
        d = dvals[i]
        x_pca_random_d = x_pca_random[1:d, :]
        x_pca_data_d = x_pca_data[1:d, :]
        wt_pca_wt_d = vec(x_pca_wt[1:d, :])

        data_random = collect_data(wt_pca_wt_d, x_pca_random_d, w_random, times)
        data_data = collect_data(wt_pca_wt_d, x_pca_data_d, counts_data, times)

        x_opt = JLD2.load(model_pars[i])["x_opt"]
        
        setting_path=joinpath(dirname(model_pars[1]), split(basename(model_pars[1]), ".")[1]*".settings.jld2")
        settings = JLD2.load(setting_path)["model_settings"]
        whiten = settings.whiten
        weight = settings.weight
        extreme = settings.extreme
        lambda = settings.lambda
        epsilon_J = settings.epsilon_J
        epsilon_sigma = settings.epsilon_sigma

        fig, ax = subplots(1, length(times), 6)
        for i in eachindex(times)
            ll_random = log_likelihood_variants(x_opt, data_random, i, lambda, epsilon_J, epsilon_sigma)
            ll_data = log_likelihood_variants(x_opt, data_data, i, lambda, epsilon_J, epsilon_sigma)
            ax[i].hist(ll_random, weights=data_random.round[i].w, bins=30, alpha=0.5, density=true, label="random samples")
            ax[i].hist(ll_data, weights=data_data.round[i].w, bins=30, alpha=0.5, density=true, label="experimental samples")
            ax[i].set_title("Round $(times[i])")
            ax[i].legend()
            ax[i].set_xlabel("log-likelihood variants")
        end

        output_name = replace(model_pars[i], "pars.jld2" => "rand_seq_log_likelihood.png")
        fig.savefig(output_name, format="png", bbox_inches="tight")
        println("Saved figure at $output_name")
        close(fig)

        #ancesor reconsruction
        n_samples_anc_reconstruction = 100
        samples_data_ancestor_reconstruction = sample(1:data_data.M, Weights(data_data.round[end].w), n_samples_anc_reconstruction; replace=false)
        samples_random_ancestor_reconstruction = sample(1:data_random.M, Weights(data_random.round[end].w), n_samples_anc_reconstruction; replace=false)
        fig_anc, ax_anc = subplots(1, length(times)-1, 6)
        for t in 1:(length(times)-1)
            ll_prob_data = transition_probability(x_opt, data_data, samples_data_ancestor_reconstruction, t, length(times), lambda, epsilon_J, epsilon_sigma)
            ll_prob_random = transition_probability(x_opt, data_random, samples_random_ancestor_reconstruction, t, length(times), lambda, epsilon_J, epsilon_sigma)

            test = SignedRankTest(ll_prob_data, ll_prob_random)
            p_values[i,t] = pvalue(test; tail=:right)

            ax_anc.hist(ll_prob_random, alpha=0.5, density=true, label="random samples")
            ax_anc.hist(ll_prob_data, alpha=0.5, density=true, label="experimental samples")    
            ax_anc.set_title("Transition round $(times[t]) to round $(times[end])")
            ax_anc.legend()
            ax_anc.set_xlabel("log-likelihood transition probability (max over ancestors)")
            ax_anc.set_ylabel("pdf")
        end

        output_name_anc = output_name = replace(model_pars[i], "pars.jld2" => "ancestor_reconstruction_likelihood.png")
        fig_anc.savefig(output_name_anc, format="png", bbox_inches="tight")
        println("Saved figure at $output_name_anc")
        close(fig_anc)
    end

    fig_p_val, ax_p_val = subplots(1, 1, 6)
    for t in 1:length(times)-1
        ax_p_val.plot(dvals, p_values[:,t], label="time $(times[t])")
    end
    ax_p_val.set_xlabel("d")
    ax_p_val.set_ylabel("p value")
    ax_p_val.legend()
    ax_p_val.set_yscale(:log)
    output_p_value = joinpath(dirname(dirname(model_pars[1])), "ancestor_reconstruction.png")
    fig_p_val.savefig(output_p_value, format="png", bbox_inches="tight")
    println("Saved figure at $output_p_value")
    close(fig_p_val)

end
