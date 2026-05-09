import Pkg
Pkg.activate("..")

using DiffusionEvolution: compute_pca, compute_norm, corr_APC, compute_frob_norm
using JLD2, PyPlot, NPZ, DelimitedFiles, LinearAlgebra, MultivariateStats, PlmDCA, PottsGauge

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)

function collect_model_score_files(base_dir::AbstractString)
    target_name = "pse1_analysis_d_*.scores.zerosumgauge_apc.tsv"

    # We mimic awk's regex capture of the d value from the full path
    re = r"/pse1_analysis_d_([0-9]+)/pse1_analysis_d_[0-9]+\.scores\.zerosumgauge_apc\.tsv$"

    pairs = Tuple{Int, String}[]  # (d, path)

    for (root, _dirs, files) in walkdir(base_dir)
        for f in files
            # quick name filter like find -name ...
            occursin(r"^pse1_analysis_d_\d+\.scores\.zerosumgauge_apc\.tsv$", f) || continue

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

file_model_scores = collect_model_score_files("/home/students/s301803/CODE/DiffusionEvolution/results/pse1/run0")
low_rank_mf_dir = "/home/students/s301803/CODE/DiffusionEvolution/results/pse1/low_rank_mf"
input_fasta = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1_clean.fasta"
wt_fasta = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/PSE1.fas"

epsilon_rel=1.0e-8
eps_warning=1.0e-4
set_zero=false

score_load = JLD2.load( "/home/students/s301803/CODE/DiffusionEvolution/results/pse1/plmdca/plmdca.score.jld2")
plmdca_score = score_load["plmdca_score"]
L = score_load["L"]
A = 21

contacts_file = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/contact_map.jld2"
true_contacts = JLD2.load(contacts_file)["contacts"]
if true_contacts != true_contacts'
    true_contacts += true_contacts'
end
true_contacts = [true_contacts[i,j]>0 ? 1 : 0 for i in 1:size(true_contacts,1), j in 1:size(true_contacts,2)]

d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

min_dist_intermediate = 12
max_dist_intermediate = 23

n_model_scores = length(file_model_scores)

n_contacts_plmdca = 0
n_positive_contacts_ou = zeros(n_model_scores)
n_positive_contacts_low_rank = zeros(n_model_scores)
n_new_contacts_low_rank_vs_plmdca = zeros(n_model_scores)
n_new_contacts_low_rank_vs_plmdca_intermediate = zeros(n_model_scores)
n_new_contacts_low_rank_vs_plmdca_long = zeros(n_model_scores)
n_new_contacts_ou_vs_plmdca = zeros(n_model_scores)
n_new_contacts_ou_vs_plmdca_intermediate = zeros(n_model_scores)
n_new_contacts_ou_vs_plmdca_long = zeros(n_model_scores)
n_new_contacts_ou_vs_low_rank = zeros(n_model_scores)
n_new_contacts_ou_vs_low_rank_intermediate = zeros(n_model_scores)
n_new_contacts_ou_vs_low_rank_long = zeros(n_model_scores)

pca = nothing

for (di, d) in pairs(d_values)
    path_score = joinpath(low_rank_mf_dir, "low_rank_mf_$d")
    global pca
    if !ispath(path_score * ".tsv")
        if isnothing(pca)
            pca = compute_pca([input_fasta], wt_fasta; weight=false, maxoutdim=A*L)
        end

        W_proj = Matrix(pca.proj[:,1:d]')
        J_low_rank = -(W_proj')*diagm(inv.(principalvars(pca)[1:d]))*W_proj
        J_low_rank_tens = permutedims(reshape(J_low_rank, A, L, A, L), (1,3,2,4))

        for i in axes(J_low_rank_tens, 3)
            for a in 1:A
                J_low_rank_tens[a,a,i,i] = 0.0
            end

            if set_zero
                J_low_rank_tens[:,:,i,i] .= 0.0
            end
        end

        frob_norm(x) = compute_frob_norm(x, L, 21)
        frobenius_norm_zerosumgauge = compute_norm(J_low_rank_tens, zeros(A,L), ZeroSumGauge(), frob_norm)
        frobenius_norm_zerosumgauge_apc = corr_APC(frobenius_norm_zerosumgauge)
        frobenius_score_zerosumgauge_apc = PlmDCA.compute_ranking(frobenius_norm_zerosumgauge_apc)

        open(path_score * ".tsv", "w") do io
            for (a,b,c) in frobenius_score_zerosumgauge_apc
                println(io, a, "\t", b, "\t", c)
            end
        end
    else
        frobenius_score_zerosumgauge_apc = [(round(Int, r[1]), round(Int, r[2]), r[3])
            for r in eachrow(readdlm(path_score * ".tsv", '\t', Float64))]
    end

    #compare predictions
    low_rank_score = frobenius_score_zerosumgauge_apc
    contacts_low_rank = []
    contacts_plmdca = []
    new_contacts_low_rank_vs_plmdca = []
    n_add_contacts_low_rank_vs_plmdca = 0
    n_add_contacts_low_rank_vs_plmdca_intermediate = 0
    n_add_contacts_low_rank_vs_plmdca_long = 0
    new_contacts_ou_vs_plmdca = []
    predictions_low_rank = []

    for i in 1:div(L,2)
        if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
            push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
        end
    end

    global n_contacts_plmdca
    if n_contacts_plmdca == 0
        n_contacts_plmdca = length(contacts_plmdca)
    end

    for i in 1:div(L,2)
        push!(predictions_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
        if true_contacts[low_rank_score[i][1], low_rank_score[i][2]] > 0
            push!(contacts_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
        end
    end

    n_add_contacts_low_rank_vs_plmdca = 0
    for c in contacts_low_rank
        if !(c in contacts_plmdca)
            n_add_contacts_low_rank_vs_plmdca += 1
            push!(new_contacts_low_rank_vs_plmdca, c)
            if min_dist_intermediate <= abs(c[1] - c[2]) <= max_dist_intermediate
                n_add_contacts_low_rank_vs_plmdca_intermediate += 1
            elseif abs(c[1] - c[2]) >= max_dist_intermediate + 1
                n_add_contacts_low_rank_vs_plmdca_long += 1
            end
        end
    end
    n_new_contacts_low_rank_vs_plmdca[di] = n_add_contacts_low_rank_vs_plmdca


    model_score = readdlm(file_model_scores[di], '\t', Float64)
    new_contacts_ou_vs_low_rank = []
    n_contacts_low_rank = length(contacts_low_rank)
    n_contacts_ou = 0
    n_add_contacts_ou_vs_low_rank = 0
    n_add_contacts_ou_vs_low_rank_intermediate = 0
    n_add_contacts_ou_vs_low_rank_long = 0
    n_add_contacts_ou_vs_plmdca = 0
    n_add_contacts_ou_vs_plmdca_intermediate = 0
    n_add_contacts_ou_vs_plmdca_long = 0
    n_agr_predictions_ou_vs_low_rank = 0
    for j in 1:div(L,2)
        score_i = round(Int, model_score[j, 1])
        score_j = round(Int, model_score[j, 2])
        if (true_contacts[score_i, score_j] > 0)
            n_contacts_ou += 1
            if !((score_i, score_j) in contacts_low_rank)
                push!(new_contacts_ou_vs_low_rank, (score_i, score_j))
                n_add_contacts_ou_vs_low_rank += 1
                if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                    n_add_contacts_ou_vs_low_rank_intermediate += 1
                elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                    n_add_contacts_ou_vs_low_rank_long += 1
                end
            end
            if !((score_i, score_j) in contacts_plmdca)
                push!(new_contacts_ou_vs_plmdca, (score_i, score_j))
                n_add_contacts_ou_vs_plmdca += 1
                if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                    n_add_contacts_ou_vs_plmdca_intermediate += 1
                elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                    n_add_contacts_ou_vs_plmdca_long += 1
                end
            end
        end
        if (score_i, score_j) in predictions_low_rank
            n_agr_predictions_ou_vs_low_rank += 1
        end
    end

    n_positive_contacts_ou[di] = n_contacts_ou
    n_positive_contacts_low_rank[di] = n_contacts_low_rank
    n_new_contacts_ou_vs_low_rank[di] = n_add_contacts_ou_vs_low_rank
    n_new_contacts_ou_vs_low_rank_intermediate[di] = n_add_contacts_ou_vs_low_rank_intermediate
    n_new_contacts_ou_vs_low_rank_long[di] = n_add_contacts_ou_vs_low_rank_long
    n_new_contacts_ou_vs_plmdca[di] = n_add_contacts_ou_vs_plmdca
    n_new_contacts_ou_vs_plmdca_intermediate[di] = n_add_contacts_ou_vs_plmdca_intermediate
    n_new_contacts_ou_vs_plmdca_long[di] = n_add_contacts_ou_vs_plmdca_long
    n_new_contacts_low_rank_vs_plmdca_intermediate[di] = n_add_contacts_low_rank_vs_plmdca_intermediate
    n_new_contacts_low_rank_vs_plmdca_long[di] = n_add_contacts_low_rank_vs_plmdca_long
end

fig_n_contacts = figure()
ax_n_contacts = gca()
msize = 3
random_baseline = (sum(true_contacts)/2)/(L*(L-1)/2)*div(L,2)
ax_n_contacts.plot(d_values, n_positive_contacts_ou, marker="o", markersize=msize, label="OU")
ax_n_contacts.plot(d_values, n_positive_contacts_low_rank, marker="o", markersize=msize, label="Low-rank")
ax_n_contacts.axhline(n_contacts_plmdca, linestyle="dashed", color="red", label="PlmDCA")
ax_n_contacts.axhline(random_baseline, linestyle="dashed", color="black", label="Random")
#ax_n_contacts.set_xticks(1:n_model_scores, [basename(f) for f in file_model_scores], rotation=90)
ax_n_contacts.legend()
ax_n_contacts.set_xlabel("d")
ax_n_contacts.set_ylabel("positive predictions")
ax_n_contacts.set_title("Precision of Contact Predictions for PSE1")
fig_n_contacts.savefig("pse1_n_contacts_comparison.svg", format="svg", bbox_inches="tight")
close(fig_n_contacts)

fig_n_new_contacts_plmdca = figure()
ax_n_new_contacts_plmdca = gca()
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca, marker="o", markersize=msize, label="OU (total)")
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca, marker="o", markersize=msize, label="Low-Rank (total)")
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca_intermediate, marker="o", markersize=msize, label="OU (intermediate-range)")
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca_long, marker="o", markersize=msize, label="OU (long-range)") 
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca_intermediate, marker="o", markersize=msize, label="Low-Rank (intermediate-range)")
ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca_long, marker="o", markersize=msize, label="Low-Rank (long-range)") 
ax_n_new_contacts_plmdca.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax_n_new_contacts_plmdca.set_xlabel("d")
ax_n_new_contacts_plmdca.set_ylabel("positive predictions")
ax_n_new_contacts_plmdca.set_title("Additional Predicted Contacts (vs PlmDCA) for PSE1")
fig_n_new_contacts_plmdca.savefig("pse1_n_new_contacts_vs_plmdca.svg", format="svg", bbox_inches="tight")
close(fig_n_new_contacts_plmdca)