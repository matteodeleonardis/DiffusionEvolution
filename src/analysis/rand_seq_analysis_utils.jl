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

function produce_random_data(avg_mut_rate, init, n_samples=1000)

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
