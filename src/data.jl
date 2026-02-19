struct Sample
    x::Matrix{Float64}
    w::Vector{Float64}
end

struct Data

    x0::Vector{Float64}
    round::Vector{Sample}
    time::Vector{Int} #[t1, ..., tN] we assume t0=0
    M::Int #number of samples
    d::Int
end

function collect_data(x0::Vector{Float64}, coordinates::Array{Float64, 3}, counts::Matrix, time::Vector{Int})

    @assert size(counts,2) == length(time) 

    w = Float64.(counts)
    if !prod(sum(w, dims=1) .≈ 1.0)
        w ./= sum(w, dims=1)
    end

    sample = Vector{Sample}(undef, size(counts, 2))
    for t in axes(coordinates, 3)
        sample[t] = Sample(coordinates[:,:,t], w[:,t])
    end

    

    return Data(x0, sample, time, size(w,2), size(coordinates, 1))
end


function collect_data(x0::Vector{Float64}, coordinates::Array{Float64, 2}, counts::Matrix, time::Vector{Int})

    @assert size(counts,2) == length(time) 

    w = Float64.(counts)
    if !prod(sum(w, dims=1) .≈ 1.0)
        w ./= sum(w, dims=1)
    end

    sample = Vector{Sample}(undef, size(counts, 2))
    for t in axes(counts, 2)
        sample[t] = Sample(coordinates, w[:,t])
    end

    

    return Data(x0, sample, time, size(w,2), size(coordinates, 1))
end


function subdata(data::Data, d::Int)

    sub_x0 = data.x0[1:d]
    sub_round = map(x->Sample(x.x[1:d,:], x.w), data.round)

    return Data(sub_x0, sub_round, data.time, data.M, d)
end


function read_sequences(file)

    is_fasta = false
    open(file, "r") do io
        first_line = readline(io)
        if startswith(first_line, ">")
            is_fasta = true
        end
    end

    if is_fasta
        return readfasta(file)
    else
        open(file, "r") do io
            lines = readlines(io)
            return map(x->("", x), lines)
        end
    end
end


function read_fasta(fasta_files::Vector{String}, fasta_file_wt=nothing)

    fasta_files = map(x->read_sequences(x), fasta_files)

    if !isnothing(fasta_file_wt)
        fasta_wt = read_sequences(fasta_file_wt)
        L = length(fasta_wt[1][2])

        println("Filtering out sequences that are not comparable with the wild-type (length=$L)")
        for i in eachindex(fasta_files)
            nseq_before = length(fasta_files[i])
            filter_length_idx = findall(map(x->length(x[2])!=L, fasta_files[i]))
            deleteat!(fasta_files[i], filter_length_idx)
            nseq_after = length(fasta_files[i])
            println("File $i/$(length(fasta_files)): $nseq_before -> $nseq_after")
            println()
        end
    
        println("Filtering out sequences with invalid symbols")
        for i in eachindex(fasta_files)
            nseq_before = length(fasta_files[i])
            filter_gap_idx = findall(map(x->sum(map(y-> occursin(y, BioSeqInt.alphabet_aa()), collect(x[2])))!=L, fasta_files[i]))
            deleteat!(fasta_files[i], filter_gap_idx)
            nseq_after = length(fasta_files[i])
            println("File $i/$(length(fasta_files)): $nseq_before -> $nseq_after")
            println()
        end
    end

    all_variants = []
    for i in eachindex(fasta_files)
        push!(all_variants, map(x->x[2], fasta_files[i]))
    end
    variants = unique(vcat(all_variants...))
    println("Unique sequences: $(length(variants))")
    counts = zeros(Int, length(variants), length(fasta_files))

    for k in eachindex(fasta_files)
        cnt_round = Dict()
        for i in eachindex(fasta_files[k])
            if haskey(cnt_round, fasta_files[k][i][2])
                cnt_round[fasta_files[k][i][2]] += 1
            else
                push!(cnt_round, fasta_files[k][i][2] => 1)
            end
        end

        for m in eachindex(variants)
            if haskey(cnt_round, variants[m])
                counts[m,k] = cnt_round[variants[m]]
            end
        end
        println("File $k/$(length(fasta_files)), unique sequences: $(length(cnt_round))")
    end

    Z = hcat(aa2int.(variants)...)
    wt = isnothing(fasta_file_wt) ? nothing : aa2int(uppercase(fasta_wt[1][2]))

    return Z, counts, wt
end

function compute_pca(fasta_files::Vector{String}, fasta_file_wt=nothing; weight, maxoutdim)
    Z, w, _ = read_fasta(fasta_files, fasta_file_wt) #fasta_file_wt provided to filter out uncompatible sequences
    x_1hot = Float64.(reshape(Flux.onehotbatch(Z, collect(1:21)), :, size(Z,2)))

    if weight
        println("Applaying weights for PCA.")
        x_1hot = transpose(transpose(x_1hot) .* w)
    end

    println("Computing PCA.")

    pca = fit(PCA, x_1hot, maxoutdim=maxoutdim);
    return pca
end

function apply_pca(pca, fasta_file_variants; whiten, extreme, d, epsilon_rel=1.0e-8)

    Z, _, _ = read_fasta(fasta_file_variants)
    x_1hot = Float64.(reshape(Flux.onehotbatch(Z, collect(1:21)), :, size(Z,2)))

    x_pca = zeros(d, size(x_1hot, 2))
    if !extreme
        x_pca .= predict(pca, x_1hot)[1:d,:]
    else
        d_large = div(d, 2) + (d%2)
        d_small = div(d, 2)
        extreme_proj = pca.proj[:,vcat(1:d_large, end-d_small+1:end)]
        x_pca .= transpose(extreme_proj) * (x_1hot .- pca.mean)
    end

    if whiten
        lambda = principalvars(pca)
        epsilon = epsilon_rel * maximum(lambda)
        x_pca .= (1.0 ./ sqrt.(lambda .+ epsilon)) .* x_pca
    end
    if size(x_pca, 2) == 1
        x_pca = dropdims(x_pca, dims=2)
    end
    return x_pca
end

function project_data(file_nat, file_wt, file_rounds; weight, maxoutdim, whiten, d, extreme)
    
    pca = compute_pca([file_nat], file_wt; weight=weight, maxoutdim=maxoutdim)
    x_pca_variants = apply_pca(pca, file_rounds; whiten=whiten, d=d, extreme=extreme)
    x_pca_wt = apply_pca(pca, [file_wt]; whiten=whiten, d=d, extreme=extreme)

    return x_pca_variants, x_pca_wt, pca
end


function process_data(file_nat, file_wt, file_rounds, times; whiten, weight=false, d, extreme=false)

    @assert length(file_rounds) == length(times) "Error: # of rounds and # of times must be equal. ($(length(file_rounds)) != $(length(times)))"
    Z, w, wt = read_fasta(file_rounds, file_wt)
    A = 21
    L = length(wt)

    if extreme
        maxoutdim = L*A
        @assert maxoutdim >= d "Error: maxoutdim must be >= d. ($(maxoutdim) < $(d))"
        x_pca_variants, x_pca_wt, pca = project_data(file_nat, file_wt, file_rounds; 
            whiten=whiten, weight=weight, maxoutdim=maxoutdim, extreme=extreme, d=d)
        return collect_data(x_pca_wt, x_pca_variants, w, times), pca
    else
        maxoutdim = d
        @assert maxoutdim >= d "Error: maxoutdim must be >= d. ($(maxoutdim) < $(d))"
        x_pca_variants, x_pca_wt, pca = project_data(file_nat, file_wt, file_rounds; 
            whiten=whiten, weight=weight, maxoutdim=maxoutdim, extreme=extreme, d=d)
        return collect_data(x_pca_wt, x_pca_variants, w, times), pca
    end
end


function data_entropy(data)

    return [-sum(x-> x==0.0 ? 0.0 : x*log(x), r.w) for r in data.round]
end