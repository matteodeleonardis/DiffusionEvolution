import Pkg
Pkg.activate("..")

using DiffusionEvolution: compare_new_contacts
using JLD2, PyPlot

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)

score_load = JLD2.load( "/home/students/s301803/CODE/DiffusionEvolution/results/pse1/ev_couplings/ev_couplings.score.jld2")
ev_couplings_score = score_load["ev_couplings_score"]
L = score_load["L"]

contacts_file = "/home/students/s301803/CODE/DiffusionEvolution/data/pse1/contact_map.jld2"
true_contacts=JLD2.load(contacts_file)["contacts"]
if true_contacts != true_contacts'
    true_contacts += true_contacts'
end
true_contacts = [true_contacts[i,j]>0 ? 1 : 0 for i in 1:size(true_contacts,1), j in 1:size(true_contacts,2)]

#comparison with plmdca
fig_plmdca, ax_plmdca = subplots(1, 1, 4)
min_dist_intermediate = 12
max_dist_intermediate = 23
plmdca_score = JLD2.load("/home/students/s301803/CODE/DiffusionEvolution/results/pse1/plmdca/plmdca.score.jld2")["plmdca_score"]
contacts_plmdca = []
for i in 1:div(L,2)
    if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
        push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
    end
end

new_contacts = []
n_contacts = 0
n_add_contacts = 0
n_add_contacts_intermediate = 0
n_add_contacts_long = 0

for j in 1:div(L,2)
    global n_contacts, n_add_contacts, n_add_contacts_intermediate, n_add_contacts_long
    score_i = round(Int, ev_couplings_score[j][1])
    score_j = round(Int, ev_couplings_score[j][2])
    if (true_contacts[score_i, score_j] > 0)
        n_contacts += 1
        if !((score_i, score_j) in contacts_plmdca)
            push!(new_contacts, (score_i, score_j))
            n_add_contacts += 1
            if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                n_add_contacts_intermediate += 1
            elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                n_add_contacts_long += 1
            end
        end
    end
end
ax_plmdca.set_title("Predicted Additional Contacts (EV Couplings vs PlmDCA) for PSE1")
ax_plmdca.set_xlabel("site i")
ax_plmdca.set_ylabel("site j")
colormap_contacts = PyPlot.matplotlib.colors.ListedColormap(["white", "grey"])
ax_plmdca.matshow(true_contacts, cmap=colormap_contacts)
ax_plmdca.scatter(map(x->x[1], new_contacts), map(x->x[2], new_contacts), color="orangered", s=5, label="correct additional predictions")
ax_plmdca.scatter([], [], color=colormap_contacts(1), label="real contact")
ax_plmdca.legend()

fig_plmdca.savefig("pse1_evcouplings_contacts_vs_plmdca.svg", format="svg", bbox_inches="tight")


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

_, _, fig_n_contacts, ax_n_contacts, _, _, _, _ = compare_new_contacts(ev_couplings_score, true_contacts, file_model_scores, div(L,2), "OU", "EVCouplings")

random_baseline = (sum(true_contacts)/2)/(L*(L-1)/2)*div(L,2)
ax_n_contacts.axhline(random_baseline, color="black", linestyle="--", label="Random")
handles, labels = ax_n_contacts.get_legend_handles_labels()
labels = ["OU", "EV Couplings", "Random"]
ax_n_contacts.legend(handles, labels)
ax_n_contacts.set_title("Precision of Contact Predictions for PSE1")


fig_n_contacts.savefig("pse1_evcouplings_n_contacts_compare.svg", format="svg", bbox_inches="tight")