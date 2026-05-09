import Pkg
Pkg.activate("..")

using MultivariateStats, JLD2, PyPlot, FastaIO, LaTeXStrings
using DiffusionEvolution: compute_J, compute_pca, derivative_nonuniform

import PyPlot.subplots
subplots(x, y ,d; kwargs...) = PyPlot.subplots(x, y; figsize=(d*y, d*x), kwargs...)

function collect_model_params_files(base_dir::AbstractString)
    target_name = "mdhfr_analysis_d_*.pars.jld2"

    # We mimic awk's regex capture of the d value from the full path
    re = r"/mdhfr_analysis_d_([0-9]+)/mdhfr_analysis_d_[0-9]+\.pars\.jld2$"

    pairs = Tuple{Int, String}[]  # (d, path)

    for (root, _dirs, files) in walkdir(base_dir)
        for f in files
            # quick name filter like find -name ...
            occursin(r"^mdhfr_analysis_d_\d+\.pars\.jld2$", f) || continue

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

input_fasta = "/home/students/s301803/CODE/DiffusionEvolution/data/dhfr/mDHFR_clean.fasta"
wt_fasta = "/home/students/s301803/CODE/DiffusionEvolution/data/dhfr/mDHFR.fasta"

A = 21
L = length(readfasta(wt_fasta)[1][2])

file_model_params = collect_model_params_files("/home/students/s301803/CODE/DiffusionEvolution/results/mdhfr/run0")
d_labels = [parse(Int, split(split(f, "/")[10], "_")[4]) for f in file_model_params]
epsilon_J = 1.0e-9

pca = nothing
off_diag_energy_tri = zeros(length(d_labels))

for i in eachindex(file_model_params)
    file = file_model_params[i]
    d = d_labels[i]
    global pca, input_fasta, wt_fasta, epsilon_J
    x_pars = JLD2.load(file)["x_opt"]
    J_inferred = compute_J(x_pars, d, epsilon_J)
    if isnothing(pca)
        pca = compute_pca([input_fasta], wt_fasta; weight=false, maxoutdim=A*L)
    end
    off_diag_energy_tri[i] = 1.0/(1.0 + sum(abs2, [J_inferred[j,j] for j in 1:d])/sum(abs2, [J_inferred[j,k] for j in 1:d, k in 1:d if j>k]))
end

off_diag_energy_tri_grad = derivative_nonuniform(d_labels, off_diag_energy_tri)
off_diag_energy_tri_grad = [(v>off_diag_energy_tri_grad[1]) ? NaN : v for v in off_diag_energy_tri_grad]

fig_odg, ax_odg = subplots(1, 1)
ax_odg.plot(d_labels, off_diag_energy_tri, marker="o")
ax_odg.set_xlabel("d")
ax_odg.set_ylabel(L"\rho")
ax_odg.set_title("Off-diagonal weight (mDHFR)")
fig_odg.savefig("dhfr_off_diag_energy_tri.svg", format="svg", bbox_inches="tight")

# fig_odg_grad, ax_odg_grad = subplots(1, 1, 6)
# ax_odg_grad.plot(d_labels, off_diag_energy_tri_grad ./ off_diag_energy_tri_grad[1], marker="o")
# ax_odg_grad.set_xlabel("d")
# ax_odg_grad.set_ylabel("rho derivative (normalized)")
# ax_odg_grad.set_title("Off-diagonal weight derivative")
# fig_odg_grad.savefig("dhfr_off_diag_energy_tri_derivative.svg", format="svg", bbox_inches="tight")
