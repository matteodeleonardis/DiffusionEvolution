import Pkg
Pkg.activate("..")

using MultivariateStats, JLD2, PyPlot

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

file_model_params = collect_model_params_files("/home/students/s301803/CODE/DiffusionEvolution/results/mdhfr/run0")
d_labels = [parse(Int, split(split(f, "/")[10], "_")[4]) for f in file_model_params]

ratio_tvar = map(file_model_params) do file
    return JLD2.load(file)["ratio_tvar"]
end

fig_var, ax_var = subplots(1, 1)
ax_var.plot(d_labels, ratio_tvar, marker="o")
ax_var.set_xlabel("d")
ax_var.set_ylabel("total variance fraction")
ax_var.set_title("Fraction of Explained Variance from mDHFR Natural Sequences")
fig_var.savefig("dhfr_cum_variance.svg", format="svg", bbox_inches="tight")