import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DiffusionEvolution: compute_pca, apply_pca
using PyPlot
using KernelDensity
import PyPlot: rc
using PyPlot: matplotlib

function main(args)

    natural = args["natural"]
    experimental = args["experimental"]
    wt = args["wt"]
    output = args["output"]

    pca = compute_pca([natural], wt; weight=false, maxoutdim=2)
    nat_x_pca = apply_pca(pca, [natural]; whiten=false, d=2, extreme=false)
    exp_x_pca = apply_pca(pca, experimental; whiten=false, d=2, extreme=false)

    plot_pca_density_overlay(nat_x_pca, exp_x_pca; pcx=1, pcy=2, outfile=output)

end

function kde_grid(x, y; n=200, xlim=nothing, ylim=nothing)
    if xlim === nothing
        xmin, xmax = minimum(x), maximum(x)
    else
        xmin, xmax = xlim
    end

    if ylim === nothing
        ymin, ymax = minimum(y), maximum(y)
    else
        ymin, ymax = ylim
    end

    kd = kde((x, y); npoints=(n, n), boundary=((xmin, xmax), (ymin, ymax)))

    # KernelDensity.jl returns kd.x, kd.y, kd.density with density[ix, iy].
    # PyPlot contour/contourf expects Z as (length(y), length(x)).
    return kd.x, kd.y, kd.density'
end

function plot_pca_density_overlay(X_nat, X_exp; pcx=1, pcy=2, outfile=nothing)
    x_nat = X_nat[pcx, :]
    y_nat = X_nat[pcy, :]
    x_exp = X_exp[pcx, :]
    y_exp = X_exp[pcy, :]

    rc("font", family="sans-serif", size=10)
    rc("axes", linewidth=0.8, labelsize=10)
    rc("xtick", labelsize=9, direction="out")
    rc("ytick", labelsize=9, direction="out")
    rc("legend", fontsize=9, frameon=false)

    fig, ax = subplots(figsize=(3.6, 3.2))

    xmin = min(minimum(x_nat), minimum(x_exp))
    xmax = max(maximum(x_nat), maximum(x_exp))
    ymin = min(minimum(y_nat), minimum(y_exp))
    ymax = max(maximum(y_nat), maximum(y_exp))

    dx = xmax - xmin
    dy = ymax - ymin
    padx = 0.03 * dx
    pady = 0.03 * dy

    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    xn, yn, zn = kde_grid(x_nat, y_nat; n=150, xlim=xlim, ylim=ylim)
    xe, ye, ze = kde_grid(x_exp, y_exp; n=150, xlim=xlim, ylim=ylim)

    znmax = maximum(zn)
    zemax = maximum(ze)

    nat_levels = range(0.05 * znmax, znmax, length=8)
    exp_levels = range(0.10 * zemax, zemax, length=6)

    ax.contourf(
        xn, yn, zn;
        levels=nat_levels,
        cmap="Greys",
        alpha=0.85,
        antialiased=true
    )

    ax.contour(
        xn, yn, zn;
        levels=nat_levels,
        colors="0.35",
        linewidths=0.4,
        alpha=0.5
    )

    ax.contourf(
        xe, ye, ze;
        levels=exp_levels,
        cmap="Reds",
        alpha=0.65,
        antialiased=true
    )

    ax.contour(
        xe, ye, ze;
        levels=exp_levels,
        colors="#a50f15",
        linewidths=0.7,
        alpha=0.8
    )

    patch_nat = matplotlib.patches.Patch(
        facecolor="0.6",
        edgecolor="0.35",
        label="Natural sequences",
        alpha=0.85
    )

    patch_exp = matplotlib.patches.Patch(
        facecolor="#fb6a4a",
        edgecolor="#a50f15",
        label="Experimental sequences",
        alpha=0.65
    )

    ax.legend(
        handles=[patch_nat, patch_exp],
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0),
        handlelength=1.2,
        handletextpad=0.4
    )

    ax.set_xlabel("PC$pcx")
    ax.set_ylabel("PC$pcy")
    ax.set_xlim(xlim...)
    ax.set_ylim(ylim...)
    ax.set_title("PCA Representation of mDHFR Variants")

    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.tick_params(length=3.5, width=0.8)

    if outfile !== nothing
        fig.savefig(outfile, format="svg", bbox_inches="tight")
    end

    return fig, ax
end

args = Dict(
    "natural" => "/home/matteo/Projects/DiffusionEvolution/data/dhfr/mDHFR_clean.fasta",
    "experimental" => [
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Round1_Q15_C10_aa.aln",
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Round2_Q15_C10_aa.aln",
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Round3_Q15_C10_aa.aln",
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Round4_Q15_C10_aa.aln",
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Round5_Q15_C10_aa.aln",
        "/home/matteo/Projects/DiffusionEvolution/data/dhfr/Gen15_aa.aln"
    ],
    "wt" => "/home/matteo/Projects/DiffusionEvolution/data/dhfr/mDHFR.fasta",
    "output" => "pca_dhfr_density.svg"
)

main(args)