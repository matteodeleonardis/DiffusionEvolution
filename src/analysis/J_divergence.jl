function J_divergence(; input_fasta, wt_fasta, output_root, file_model_scores, epsilon_J=1.0e-9)

    L = length(readfasta(wt_fasta)[1][2])
    A=21
    pca = nothing
    d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

    J_div_score = zeros(length(d_values))
    off_diag_energy = zeros(length(d_values))
    off_diag_energy_tri = zeros(length(d_values))
    spectrum_dev = zeros(length(d_values))

    for (di, d) in pairs(d_values)
        path_pars = joinpath(dirname(file_model_scores[di]), split(basename(file_model_scores[di]), ".")[1]*".pars.jld2")
        x_pars = JLD2.load(path_pars)["x_opt"]
        J_inferred = compute_J(x_pars, d, epsilon_J)
        if isnothing(pca)
            pca = compute_pca([input_fasta], wt_fasta; weight=false, maxoutdim=A*L)
        end

        J_div_score[di] = sum(abs2, J_inferred - diagm(inv.(principalvars(pca)[1:d])))/length(J_inferred)

        off_diag_energy[di] = sum(abs2, J_inferred - diagm(diag(J_inferred)))/sum(abs2, J_inferred)

        off_diag_energy_tri[di] = 1.0/(1.0 + sum(abs2, [J_inferred[i,i] for i in 1:d])/sum(abs2, [J_inferred[i,j] for i in 1:d, j in 1:d if i>j]))

        spectrum_J = eigen(J_inferred).values
        spectrum_dev[di] = sum(abs2, log.(sort(spectrum_J)) - log.(sort(inv.(principalvars(pca)[1:d]))))/d


    end

    J_div_score_grad = derivative_nonuniform(d_values, J_div_score)
    J_div_score_grad = [(v<J_div_score_grad[1]) ? NaN : v for v in J_div_score_grad]
    off_diag_energy_grad = derivative_nonuniform(d_values, off_diag_energy)
    off_diag_energy_grad = [(v>off_diag_energy_grad[1]) ? NaN : v for v in off_diag_energy_grad]
    off_diag_energy_tri_grad = derivative_nonuniform(d_values, off_diag_energy_tri)
    off_diag_energy_tri_grad = [(v>off_diag_energy_tri_grad[1]) ? NaN : v for v in off_diag_energy_tri_grad]
    spectrum_dev_grad = derivative_nonuniform(d_values, spectrum_dev)
    spectrum_dev_grad = [(v<spectrum_dev_grad[1]) ? NaN : v for v in spectrum_dev_grad]

    fig_Jdiv = figure()
    ax_Jdiv = gca()
    ax_Jdiv.plot(d_values, J_div_score, marker="o")
    ax_Jdiv.set_xlabel("d")
    ax_Jdiv.set_ylabel("value")
    ax_Jdiv.set_title("J divergence")
    fig_Jdiv.savefig(output_root * ".J_divergence.svg", format="svg", bbox_inches="tight")
    close(fig_Jdiv)

    fig_Jdiv_grad = figure()
    ax_Jdiv_grad = gca()
    ax_Jdiv_grad.plot(d_values, J_div_score_grad ./ J_div_score_grad[1], marker="o")
    ax_Jdiv_grad.set_xlabel("d")
    ax_Jdiv_grad.set_ylabel("value (normalized)")
    ax_Jdiv_grad.set_title("J divergence derivative")
    fig_Jdiv_grad.savefig(output_root * ".J_divergence_derivative.svg", format="svg", bbox_inches="tight")
    close(fig_Jdiv_grad)

    fig_Jode = figure()
    ax_Jode = gca()
    ax_Jode.plot(d_values, off_diag_energy, marker="o")
    ax_Jode.set_xlabel("d")
    ax_Jode.set_ylabel("value")
    ax_Jode.set_title("J off-diagonal energy")
    fig_Jode.savefig(output_root * ".J_off_diagonal_energy.svg", format="svg", bbox_inches="tight")
    close(fig_Jode)

    fig_Jodet = figure()
    ax_Jodet = gca()
    ax_Jodet.plot(d_values, off_diag_energy_tri, marker="o")
    ax_Jodet.set_xlabel("d")
    ax_Jodet.set_ylabel("value")
    ax_Jodet.set_title("J off-diagonal energy (triangular)")
    fig_Jodet.savefig(output_root * ".J_off_diagonal_energy_tri.svg", format="svg", bbox_inches="tight")
    close(fig_Jodet)

    fig_Jode_grad = figure()
    ax_Jode_grad = gca()
    ax_Jode_grad.plot(d_values, off_diag_energy_grad ./ off_diag_energy_grad[1], marker="o")
    ax_Jode_grad.set_xlabel("d")
    ax_Jode_grad.set_ylabel("value (normalized)")
    ax_Jode_grad.set_title("J off-diagonal energy derivative")
    fig_Jode_grad.savefig(output_root * ".J_off_diagonal_energy_derivative.svg", format="svg", bbox_inches="tight")
    close(fig_Jode_grad)

    fig_specdev = figure()
    ax_specdev = gca()
    ax_specdev.plot(d_values, spectrum_dev, marker="o")
    ax_specdev.set_xlabel("d")
    ax_specdev.set_ylabel("value")
    ax_specdev.set_title("J spectrum deviation")
    fig_specdev.savefig(output_root * ".J_spectrum_deviation.svg", format="svg", bbox_inches="tight")
    close(fig_specdev)

    fig_specdev_grad = figure()
    ax_specdev_grad = gca()
    ax_specdev_grad.plot(d_values, spectrum_dev_grad ./ spectrum_dev_grad[1], marker="o")
    ax_specdev_grad.set_xlabel("d")
    ax_specdev_grad.set_ylabel("value (normalized)")
    ax_specdev_grad.set_title("J spectrum deviation derivative")
    fig_specdev_grad.savefig(output_root * ".J_spectrum_deviation_derivative.svg", format="svg", bbox_inches="tight")
    close(fig_specdev_grad)

end



        