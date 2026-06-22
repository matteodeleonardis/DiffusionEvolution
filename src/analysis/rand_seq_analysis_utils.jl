function compute_average_mut_rate(input::Vector, init)
    
    Z, counts, wt = read_fasta(input, init)
    L = length(wt)
    dist_from_wt = [zeros(Int, size(Z, 2)) for _ in eachindex(input)]
    avg_mut_rate = zeros(length(input))
    for i in eachindex(input)
        for s in axes(Z, 2)
            dist_from_wt[i][s] = sum(Z[:,s] .!= wt)
        end
    end
    for i in eachindex(input)
        avg_mut_rate[i] = sum(counts[:,i] .* dist_from_wt[i] ./ L) / sum(counts[:,i])
    end

    return avg_mut_rate, dist_from_wt, counts
end

function produce_random_data(avg_mut_rate, init; n_samples=1000)

    init_seq = readfasta(init)[1][2]
    L = length(init_seq)

    rand_samples = [Vector{String}(undef, n_samples) for i in eachindex(avg_mut_rate)]
    dist_from_wt = [zeros(Int, n_samples) for i in eachindex(avg_mut_rate)]

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
            dist_from_wt[i][s] = sum(collect(rand_seq) .!= collect(init_seq))
        end
    end

    return rand_samples, dist_from_wt
end

function compute_profile_stats(input::Vector, init)
    
    Z, counts, wt = read_fasta(input, init)
    L = length(wt)
    dist_from_wt = [zeros(Int, size(Z, 2)) for _ in eachindex(input)]
    fi = zeros(length(input), L, 21)
    for i in eachindex(input)
        for s in axes(Z, 2)
            dist_from_wt[i][s] = sum(Z[:,s] .!= wt)
            for pos in axes(Z, 1)
                aa = Z[pos, s]
                fi[i, pos, aa] += counts[s, i]
            end
        end
        
    end
    fi ./= sum(fi, dims=3)

    return fi, dist_from_wt, counts
end

function produce_random_profile_data(fi, init; n_samples=1000)

    init_seq = readfasta(init)[1][2]
    L = length(init_seq)

    rand_samples = [Vector{String}(undef, n_samples) for i in axes(fi, 1)]
    dist_from_wt = [zeros(Int, n_samples) for i in axes(fi,1)]

    for i in axes(fi, 1)
        println("Generating random samples $i/$(size(fi, 1))")
        for s in 1:n_samples
            seq = zeros(Int, L)
            for pos in 1:L
                seq[pos] = sample(1:21, Weights(fi[i, pos, :]))
            end
            rand_seq = String(int2aa.(seq))
            rand_samples[i][s] = rand_seq
            dist_from_wt[i][s] = sum(collect(rand_seq) .!= collect(init_seq))
        end
    end

    return rand_samples, dist_from_wt
end

function compute_site_mut_stats(input::Vector, init)
    
    Z, counts, wt = read_fasta(input, init)
    L = length(wt)
    dist_from_wt = [zeros(Int, size(Z, 2)) for _ in eachindex(input)]
    fmut = zeros(length(input), L)
    for i in eachindex(input)
        for s in axes(Z, 2)
            dist_from_wt[i][s] = sum(Z[:,s] .!= wt)
            for pos in axes(Z, 1)
                aa = Z[pos, s]
                if aa != wt[pos]
                    fmut[i, pos] += counts[s, i]
                end
            end
        end
        
    end
    fmut ./= transpose(sum(counts, dims=1))

    return fmut, dist_from_wt, counts
end

function produce_site_mut_data(fmut, init; n_samples=1000)

    init_seq = readfasta(init)[1][2]
    L = length(init_seq)

    rand_samples = [Vector{String}(undef, n_samples) for i in axes(fmut, 1)]
    dist_from_wt = [zeros(Int, n_samples) for i in axes(fmut,1)]

    for i in axes(fmut, 1)
        println("Generating random samples $i/$(size(fmut, 1))")
        for s in 1:n_samples
            seq = collect(init_seq)
            for pos in 1:L
                if rand() < fmut[i, pos]
                    seq[pos] = rand(collect(replace(alphabet_aa(), "-" => "", seq[pos] => "")))
                end
            end
            rand_samples[i][s] = String(seq)
            dist_from_wt[i][s] = sum(collect(seq) .!= collect(init_seq))
        end
    end

    return rand_samples, dist_from_wt
end


function collect_samples_data(input, init, file_nat, rand_samples, d_max)

    @assert length(input) == length(rand_samples)
    n_times = length(rand_samples)
    # random samples
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

    # experimental samples
    Z_data, counts_data, wt_data = read_fasta(input, init)

    pca = compute_pca([file_nat], init; weight=false, maxoutdim=d_max)

    return Z_random, w_random, Z_data, counts_data, wt_data, pca
end

function compare_pca_moments(data_data, data_random, times, output_root)

    PyPlot.matplotlib.rcParams["svg.fonttype"] = "none"

    n_times = length(times)

    fig_mean, ax_mean = subplots(1, n_times, 6)
    fig_var, ax_var = subplots(1, n_times, 6)
    fig_cov, ax_cov = subplots(1, n_times, 6)

    for t in eachindex(times)

        x_data = data_data.round[t].x
        x_rand = data_random.round[t].x

        w_data = data_data.round[t].w
        w_rand = data_random.round[t].w

        mu_data = dropdims(mean(x_data, Weights(w_data), dims=2), dims=2)
        mu_rand = dropdims(mean(x_rand, Weights(w_rand), dims=2), dims=2)

        cov_data = cov(x_data, Weights(w_data), 2)
        cov_rand = cov(x_rand, Weights(w_rand), 2)

        var_data = diag(cov_data)
        var_rand = diag(cov_rand)

        mean_corr = cor(mu_data, mu_rand)
        var_corr = cor(var_data, var_rand)
        cov_corr = cor(vec(cov_data), vec(cov_rand))

        mean_dist = norm(mu_data - mu_rand)
        cov_rel_dist = norm(cov_data - cov_rand) / norm(cov_data)

        # Mean comparison
        ax_mean[t].scatter(mu_data, mu_rand, alpha=0.7)

        lo = minimum(vcat(mu_data, mu_rand))
        hi = maximum(vcat(mu_data, mu_rand))
        ax_mean[t].plot([lo, hi], [lo, hi], linestyle="--")

        ax_mean[t].set_xlabel("experimental mean PC")
        ax_mean[t].set_ylabel("profile-random mean PC")
        ax_mean[t].set_title("t=$(times[t])")

        ax_mean[t].text(
            0.05, 0.95,
            "ρ = $(round(mean_corr, digits=3))\n‖Δμ‖ = $(round(mean_dist, sigdigits=3))",
            transform=ax_mean[t].transAxes,
            verticalalignment="top",
            bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.8)
        )

        # Variance comparison
        ax_var[t].scatter(var_data, var_rand, alpha=0.7)

        lo = minimum(vcat(var_data, var_rand))
        hi = maximum(vcat(var_data, var_rand))
        ax_var[t].plot([lo, hi], [lo, hi], linestyle="--")

        ax_var[t].set_xlabel("experimental PC variance")
        ax_var[t].set_ylabel("profile-random PC variance")
        ax_var[t].set_title("t=$(times[t])")

        ax_var[t].text(
            0.05, 0.95,
            "ρ = $(round(var_corr, digits=3))",
            transform=ax_var[t].transAxes,
            verticalalignment="top",
            bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.8)
        )

        # Full covariance comparison
        ax_cov[t].scatter(vec(cov_data), vec(cov_rand), alpha=0.3)

        lo = minimum(vcat(vec(cov_data), vec(cov_rand)))
        hi = maximum(vcat(vec(cov_data), vec(cov_rand)))
        ax_cov[t].plot([lo, hi], [lo, hi], linestyle="--")

        ax_cov[t].set_xlabel("experimental covariance entries")
        ax_cov[t].set_ylabel("profile-random covariance entries")
        ax_cov[t].set_title("t=$(times[t])")

        ax_cov[t].text(
            0.05, 0.95,
            "ρ = $(round(cov_corr, digits=3))\nrel. dist = $(round(cov_rel_dist, sigdigits=3))",
            transform=ax_cov[t].transAxes,
            verticalalignment="top",
            bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.8)
        )
    end

    fig_mean.savefig(output_root * ".pca_mean_comparison.png", format="png", bbox_inches="tight")
    fig_var.savefig(output_root * ".pca_variance_comparison.png", format="png", bbox_inches="tight")
    fig_cov.savefig(output_root * ".pca_covariance_comparison.png", format="png", bbox_inches="tight")

    close(fig_mean)
    close(fig_var)
    close(fig_cov)

    return nothing
end