import Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))

using DiffusionEvolution
using DelimitedFiles
using PyPlot

# Input p-value txt file.
# You can pass it from the command line:
#
#   julia script.jl results/mdhfr/run0_pfam/forward_ancestor_reconstruction_pvalue_site_mut.txt
#
# or edit this default path:
input_txt = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "../../../results/pse1/run0_lastout/forward_ancestor_reconstruction_pvalue_uniform.txt")

model_label = split(input_txt, "/")[end-1]

# Save in the same folder as this script, with the same name as the txt file.
output_svg = joinpath(
    @__DIR__,
    model_label * "_" * splitext(basename(input_txt))[1] * "_significance_map.svg",
)

raw = readdlm(input_txt)

dvals = sort(unique(Int.(raw[:, 1])))
time_labels = sort(unique(Int.(raw[:, 2])))

p_values = fill(NaN, length(dvals), length(time_labels))

for row in axes(raw, 1)
    d = Int(raw[row, 1])
    t = Int(raw[row, 2])

    i = findfirst(==(d), dvals)
    j = findfirst(==(t), time_labels)

    p_values[i, j] = Float64(raw[row, 3])
end

# Significance thresholds
alpha = 1e-3


# Code each point as:
# 0 = not significant
# 1 = significant at p < 0.05
# 2 = significant after Bonferroni correction
significance = zeros(Int, size(p_values))

for i in axes(p_values, 1), j in axes(p_values, 2)
    p = p_values[i, j]

    if isfinite(p)
        if p < alpha
            significance[i, j] = 1
        end
    end
end

# Keep SVG smaller: do not convert text to paths.
PyPlot.rc("svg", fonttype="none")

fig, ax = PyPlot.subplots(1, 1; figsize=(5.0, 2.6))

# Heatmap: rows are times, columns are d values.
img = ax.imshow(
    significance',
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=2,
)

# Axis ticks
ax.set_xticks(0:length(dvals)-1)
ax.set_xticklabels(string.(dvals), rotation=45, ha="right", fontsize=7)

ax.set_yticks(0:length(time_labels)-1)
ax.set_yticklabels(string.(time_labels), fontsize=8)

ax.set_xlabel("latent dimension d")
ax.set_ylabel("ancestor time")
ax.set_title("Forward ancestor reconstruction significance")

# Colorbar
cbar = fig.colorbar(img, ax=ax, ticks=[0, 1])
cbar.ax.set_yticklabels([
    "not significant",
    "p < $alpha",
])
cbar.ax.tick_params(labelsize=7)

# Optional: draw grid lines between cells
ax.set_xticks(collect(-0.5:1:length(dvals)-0.5), minor=true)
ax.set_yticks(collect(-0.5:1:length(time_labels)-0.5), minor=true)
ax.grid(which="minor", linewidth=0.4)
ax.tick_params(which="minor", bottom=false, left=false)

fig.tight_layout()
fig.savefig(output_svg, format="svg", bbox_inches="tight")
close(fig)

println("Saved figure at: ", output_svg)