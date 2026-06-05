function compute_average_mut_rate(input::Vector, init)
    
    Z, counts, wt = read_fasta(input, init)
    L = length(wt)
    dist_from_wt = zeros(Int, size(Z, 2))
    avg_mut_rate = zeros(length(input))
    for s in axes(Z, 2)
        dist_from_wt[s] = sum(Z[:,s] .!= wt)
    end
    for i in eachindex(input)
        avg_mut_rate[i] = sum(counts[:,i] .* dist_from_wt ./ L) / sum(counts[:,i])
    end

    return avg_mut_rate
end

function produce_random_data(avg_mut_rate, init, n_samples=1000)

    init_seq = readfasta(init)[1][2]
    L = length(init_seq)

    rand_samples = [Vector{String}(undef, n_samples) for i in eachindex(avg_mut_rate)]

    for i in eachindex(avg_mut_rate)
        println("Generating random samples $i/$(length(avg_mut_rate)))")
        mut_rate = avg_mut_rate[i]
        for s in 1:n_samples
            n_muts = rand(Binomial(L, mut_rate))
            mut_positions = randperm(L)[1:n_muts]
            rand_seq = collect(init_seq)
            for pos in mut_positions
                rand_seq[pos] = rand(collect(replace(alphabet_aa(), "-" => "", rand_seq[pos] => "")))
            end
            rand_samples[i][s] = String(rand_seq)
        end
    end

    return rand_samples
end

function collect_samples_data(input, init, file_nat, rand_samples, d_max)

    @assert length(input) == length(rand_samples)
     n_times = length(rand_samples)
    #random samples
    random_variants_counts = Dict()
   
    for i in eachindex(rand_samples)
        for s in eachindex(rand_samples[i])
            if haskey(random_variants_counts, rand_samples[i][s])
                random_variants_counts[rand_samples[i][s]][i] += 1
            else
                push!(random_variants_counts, rand_samples[i][s] => zeros(Int, n_times))
                random_variants_counts[rand_samples[i][s]][i] = 1
            end
        end
    end

    random_variants = unique(vcat(rand_samples...))
    w_random = zeros(length(random_variants), n_times)
    for i in eachindex(random_variants)
        w_random[i, :] = random_variants_counts[random_variants[i]]
    end
    w_random = w_random ./ sum(w_random, dims=1)
    Z_random = hcat(aa2int.(random_variants)...)

    #experimental samples
    Z_data, counts_data, wt_data = read_fasta(input, init)

    pca = compute_pca([file_nat], init; weight=false, maxoutdim=d_max)

    return Z_random, w_random, Z_data, counts_data, wt_data, pca
end


function rand_seq_analysis_pse1(file_wt, file_rounds, file_nat, model_pars)

    times = [10, 20]

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
    end

end