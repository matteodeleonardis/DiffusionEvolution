function run_plmdca_analysis_pse1(;input_fasta, contacts_file, output_root, fixed, file_model_scores::Vector)

    if !ispath(output_root * ".score.jld2")
        output_plmdca = plmdca(input_fasta; min_separation=5)
        _, L = size(output_plmdca.htensor)
        plmdca_score = output_plmdca.score
        @save output_root * ".score.jld2" plmdca_score L
    else
        println("Loading plmdca scores.")
        score_load = JLD2.load(output_root * ".score.jld2")
        plmdca_score = score_load["plmdca_score"]
        L = score_load["L"]
    end

    true_contacts=JLD2.load(contacts_file)["contacts"]
    if true_contacts != true_contacts'
        true_contacts += true_contacts'
    end
    ppv_plmdca = compute_true_positives(plmdca_score, true_contacts, x -> x>0.0)

    #ppv curve
    figure()
    plot(ppv_plmdca[1:L], label="plmdca")
    xticks([0, L÷2, L], ["0", "L/2", "L"])
    legend()
    gcf().savefig(output_root * ".ppv.png", format="png", bbox_inches="tight")

    #contact plot
    figure()
    ax=gca()
    contact_plot(plmdca_score, true_contacts, div(L,2), ax=ax)
    ax.set_xlabel("site i")
    ax.set_ylabel("site j")
    gcf().savefig(output_root * ".contact.png", format="png", bbox_inches="tight")
    
    #new contacts plot
    fig, ax, fig_n_contacts, ax_n_contacts, fig_acc, ax_acc = compare_new_contacts(plmdca_score, true_contacts, file_model_scores, div(L,2))
    fig.savefig(output_root * ".new_contacts.png", format="png", bbox_inches="tight")
    fig_n_contacts.savefig(output_root * ".n_contacts.png", format="png", bbox_inches="tight")
    fig_acc.savefig(output_root * ".prediction_agreement.png", format="png", bbox_inches="tight")

    #gamma plot
    output_files=[joinpath(dirname(f), split(basename(f), ".")[1]) for f in file_model_scores]
    if fixed == false
        fig_gamma, ax_gamma, gammas, gamma_est = plot_gamma(output_files)
        fig_gamma.savefig(output_root * ".gamma.png", format="png", bbox_inches="tight")
        out_gamma = open(output_root * ".gamma.txt", "w")
        print(out_gamma, "\t")
        println(out_gamma, join(output_files, "\t"))
        println(out_gamma, "gamma inferred: \t", join(string.(gammas), "\t"))
        println(out_gamma, "gamma empirical: \t", join(string.(gamma_est), "\t"))
        close(out_gamma)
    end

    #ratio_tvar plot
    dlabel = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]
    ratio_tvar = map(file_model_scores) do f
        file_par = split(basename(f), ".")[1] * ".pars.jld2"
        r = JLD2.load(joinpath(dirname(f), file_par))["ratio_tvar"]
        return r
    end
    fig_rtvar = figure()
    ax_rtvar = gca()
    ax_rtvar.plot(dlabel, ratio_tvar, marker="o")
    fig_rtvar.savefig(output_root * ".ratio_tvar.png", format="png", bbox_inches="tight")

    #ppv comparison
    fig_ppv, ax_ppv = subplots(1, length(file_model_scores), 6)
    for i in eachindex(file_model_scores)
        scores_d = []
        open(file_model_scores[i], "r") do io
            for line in eachline(io)
                si, sj, score = split(line)
                push!(scores_d, (parse(Int, si), parse(Int, sj), parse(Float64, score)))
            end
            ppv_d = compute_true_positives(scores_d, true_contacts, x -> x>0.0)
            ax_ppv[i].plot(ppv_plmdca[1:L], label="PlmDCA")
            ax_ppv[i].plot(ppv_d[1:L], label="OU model")
            ax_ppv[i].set_xticks([0, L÷2, L], ["0", "L/2", "L"])
            ax_ppv[i].set_ylabel("positive prediction fraction")
            ax_ppv[i].set_title("PPV curve (d=$(dlabel[i]))")
            ax_ppv[i].set_ylim(0.0, 1.05)
            ax_ppv[i].legend()
        end
    end
    fig_ppv.savefig(output_root * ".ppv_compare.png", format="png", bbox_inches="tight")

    #contact plot comparison
    fig_cont, ax_cont = subplots(1, length(file_model_scores), 6)
    for i in eachindex(file_model_scores)
        scores_d = []
        open(file_model_scores[i], "r") do io
            for line in eachline(io)
                si, sj, score = split(line)
                push!(scores_d, (parse(Int, si), parse(Int, sj), parse(Float64, score)))
            end
            contact_plot(plmdca_score, true_contacts, div(L,2); color_pos="purple", ax=ax_cont[i], flip=true)
            contact_plot(scores_d, true_contacts, div(L,2); color_pos="blue", ax=ax_cont[i])
            ax_cont[i].set_xlabel("site i")
            ax_cont[i].set_ylabel("site j")
            ax_cont[i].set_title("Contact Map Predictions (d=$(dlabel[i]))")

            ax_cont[i].scatter([], [], color="purple", label="positive prediction PlmDCA")
            ax_cont[i].scatter([], [], color="blue", label="postive prediction OU model")
            ax_cont[i].scatter([], [], color="red", label="incorrect prediction")
            ax_cont[i].legend(loc="lower left", bbox_to_anchor=(1.05, 0.0))
        end
    end
    fig_cont.savefig(output_root * ".contact_map_compare.png", format="png", bbox_inches="tight")

end