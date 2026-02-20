function J_divergence(; input_fasta, wt_fasta, output_root, file_model_scores, epsilon_J=1.0e-9)

    L = length(readfasta(wt_fasta)[1][2])
    A=21
    pca = nothing
    d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

    J_div_score = zeros(length(d_values))
    off_diag_energy = zeros(length(d_values))
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

        spectrum_J = eigen(J_inferred).values
        spectrum_dev[di] = sum(abs2, log.(sort(spectrum_J)) - log.(sort(inv.(principalvars(pca)[1:d]))))/d


    end

    fig_Jdiv = figure()
    ax_Jdiv = gca()
    ax_Jdiv.plot(d_values, J_div_score)
    ax_Jdiv.set_xlabel("d")
    ax_Jdiv.set_ylabel("value")
    ax_Jdiv.set_title("J divergence")
    fig_Jdiv.savefig(output_root * ".J_divergence.png", format="png", bbox_inches="tight")
    close(fig_Jdiv)

    fig_Jode = figure()
    ax_Jode = gca()
    ax_Jode.plot(d_values, off_diag_energy)
    ax_Jode.set_xlabel("d")
    ax_Jode.set_ylabel("value")
    ax_Jode.set_title("J off-diagonal energy")
    fig_Jode.savefig(output_root * ".J_off_diagonal_energy.png", format="png", bbox_inches="tight")
    close(fig_Jode)

    fig_specdev = figure()
    ax_specdev = gca()
    ax_specdev.plot(d_values, spectrum_dev)
    ax_specdev.set_xlabel("d")
    ax_specdev.set_ylabel("value")
    ax_specdev.set_title("J spectrum deviation")
    fig_specdev.savefig(output_root * ".J_spectrum_deviation.png", format="png", bbox_inches="tight")
    close(fig_specdev)

end



        