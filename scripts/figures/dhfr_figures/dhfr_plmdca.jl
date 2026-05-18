import Pkg
Pkg.activate("..")

using DiffusionEvolution: compute_true_positives, contact_plot
using JLD2, PyPlot, NPZ

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)

score_load = JLD2.load( "/home/students/s301803/CODE/DiffusionEvolution/results/mdhfr/plmdca/plmdca.score.jld2")
plmdca_score = score_load["plmdca_score"]
L = score_load["L"]

contacts_file = "/home/students/s301803/CODE/DiffusionEvolution/data/dhfr/contact_map.npy"
true_contacts = npzread(contacts_file)
true_contacts = [true_contacts[i,j]>0 ? 1 : 0 for i in 1:size(true_contacts,1), j in 1:size(true_contacts,2)]
ppv_plmdca = compute_true_positives(plmdca_score, true_contacts, x -> x>0.0)

random_baseline = (sum(true_contacts)/2)/(L*(L-1)/2)

#ppv curve
fig_ppv, ax_ppv = subplots(1, 1, 4)
ax_ppv.plot(ppv_plmdca[1:L], label="PlmDCA")
ax_ppv.axhline(random_baseline, color="red", linestyle="--", label="Random")
ax_ppv.set_title("mDHFR PPV Curve for PlmDCA Contact Predictions")
ax_ppv.set_ylim(0, 1.05)
ax_ppv.set_xticks([0, L÷2, L])
ax_ppv.set_xticklabels(["0", "L/2", "L"])
ax_ppv.set_xlabel("number of pairs")
ax_ppv.set_ylabel("fraction correct predictions")
ax_ppv.legend()
fig_ppv.savefig("mdhfr_plmdca_ppv.svg", format="svg", bbox_inches="tight")

#contact plot
fig_contact, ax_contact = subplots(1, 1, 4)
cmap_contact = PyPlot.matplotlib.colors.ListedColormap(["white", "grey"])
contact_plot(plmdca_score, true_contacts, div(L,2), cmap=cmap_contact, ax=ax_contact)
color_contact = cmap_contact(1.0)
ax_contact.set_xlabel("site i")
ax_contact.set_xticks([], [])
ax_contact.set_ylabel("site j")
ax_contact.set_yticks([], [])
ax_contact.scatter([], [], color="blue", label="correct prediction")
ax_contact.scatter([], [], color="red", label="incorrect prediction")
ax_contact.scatter([], [], color=color_contact, label="real contact")
ax_contact.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
ax_contact.set_title("mDHFR Contact Map Predictions for PlmDCA")
fig_contact.savefig("mdhfr_plmdca_contact.svg", format="svg", bbox_inches="tight")
