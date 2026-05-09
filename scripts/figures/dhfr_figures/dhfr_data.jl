import Pkg
Pkg.activate("../")

using DiffusionEvolution: process_data
using PyPlot, JLD2, FastaIO

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)


#load data
data_dir = joinpath(@__DIR__, "../../../data")

file0 = joinpath(data_dir, "dhfr/mDHFR.fasta")
file1 = joinpath(data_dir, "dhfr/Round1_Q15_C10_aa.aln")
file2 = joinpath(data_dir, "dhfr/Round2_Q15_C10_aa.aln")
file3 = joinpath(data_dir, "dhfr/Round3_Q15_C10_aa.aln")
file4 = joinpath(data_dir, "dhfr/Round4_Q15_C10_aa.aln")
file5 = joinpath(data_dir, "dhfr/Round5_Q15_C10_aa.aln")
file15 = joinpath(data_dir, "dhfr/Gen15_aa.aln")

times = [1,2,3,4,5, 15]

file_nat = joinpath(data_dir, "dhfr/mDHFR_clean.fasta")
file_rounds = [file1, file2, file3, file4, file5, file15]

data, pca = process_data(file_nat, file0, file_rounds, times; whiten=false, weight=false,
                                d=2, extreme=false)

fig, ax = subplots(1, length(times), 4)
for i in 1:length(times)
    fig_hist = ax[i].hist2d(data.round[i].x[1,:], data.round[i].x[2,:], weights=data.round[i].w, bins=50)
    fig_hist[4].set_edgecolor("face")
    fig_hist[4].set_rasterized(true)
    ax[i].set_title("Round $(times[i])")
    ax[i].set_xlabel("PC1")
    ax[i].set_ylabel("PC2")
    ax[i].scatter(data.x0[1], data.x0[2], color="red", label="Wild-Type", marker= "d", s=100, edgecolor="black")

    if i == length(times)
        fig.colorbar(fig_hist[4], ax=ax)
    end
end

fig.suptitle("mDHFR Latent Representation")

fig.savefig("dhfr_latent_representation.svg", format="svg", bbox_inches="tight")
