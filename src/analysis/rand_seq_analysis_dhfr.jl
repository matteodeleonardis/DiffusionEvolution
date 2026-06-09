function rand_seq_analysis_dhfr(file_wt, file_rounds, file_nat, model_pars)

    times = [1,2,3,4,5,15]

    dvals = [parse(Int, split(split(basename(p), ".")[1], "_")[end]) for p in model_pars]
    dmax = maximum(dvals)

    avg_mut_rate = compute_average_mut_rate(file_rounds, file_wt)
    rand_samples = produce_random_data(avg_mut_rate, file_wt)
    Z_random, w_random, Z_data, counts_data, wt_data, pca = collect_samples_data(file_rounds, file_wt, file_nat, rand_samples, dmax)

    x_1hot_wt = Float64.(reshape(Flux.onehotbatch(wt_data, collect(1:21)), :, 1))
    x_pca_wt = predict(pca, x_1hot_wt)
    
    x_1hot_random = Float64.(reshape(Flux.onehotbatch(Z_random, collect(1:21)), :, size(Z_random,2)))
    x_pca_random = predict(pca, x_1hot_random)

    x_1hot_data = Float64.(reshape(Flux.onehotbatch(Z_data, collect(1:21)), :, size(Z_data,2)))
    x_pca_data = predict(pca, x_1hot_data)
    

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

        #random sequences log-likelihood
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

        output_name = output_name = replace(model_pars[i], "pars.jld2" => "rand_seq_log_likelihood.png")
        fig.savefig(output_name, format="png", bbox_inches="tight")
        println("Saved figure at $output_name")
        close(fig)

        #ancesor reconsruction
        n_samples_anc_reconstruction = 100
        samples_data_ancestor_reconstruction = randperm(data_data.M)[1:n_samples_anc_reconstruction]
        samples_random_ancestor_reconstruction = randperm(data_random.M)[1:n_samples_anc_reconstruction]
        fig_anc, ax_anc = subplots(1, length(times)-1, 6)
        for i in 1:(length(times)-1)
            ll_prob_data = transition_probability(x_opt, data_data, samples_data_ancestor_reconstruction, i, length(times), lambda, epsilon_J, epsilon_sigma)
            ll_prob_random = transition_probability(x_opt, data_random, samples_random_ancestor_reconstruction, i, length(times), lambda, epsilon_J, epsilon_sigma)

            ax_anc[i].hist(ll_prob_random, alpha=0.5, density=true, label="random samples")
            ax_anc[i].hist(ll_prob_data, alpha=0.5, density=true, label="experimental samples")    
            ax_anc[i].set_title("Transition round $(times[i]) to round $(times[end])")
            ax_anc[i].legend()
            ax_anc[i].set_xlabel("log-likelihood transition probability (max over ancestors)")
            ax_anc[i].set_ylabel("pdf")
        end

        output_name_anc = output_name = replace(model_pars[i], "pars.jld2" => "ancestor_reconstruction_likelihood.png")
        fig_anc.savefig(output_name_anc, format="png", bbox_inches="tight")
        println("Saved figure at $output_name_anc")
        close(fig_anc)
    end

end
