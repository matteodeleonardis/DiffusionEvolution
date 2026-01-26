function run_plmdca_analysis_dhfr(;input_fasta, contacts_file, output_root, file_model_scores::Vector)

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

    true_contacts = npzread(contacts_file)
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
    contact_plot(plmdca_score, true_contacts, L, ax=ax)
    ax.set_xlabel("site i")
    ax.set_ylabel("site j")
    gcf().savefig(output_root * ".contact.png", format="png", bbox_inches="tight")
    
    #new contacts plot
    fig, ax, fig_n_contacts, ax_n_contacts = compare_new_contacts(plmdca_score, true_contacts, file_model_scores, div(L,2))
    fig.savefig(output_root * ".new_contacts.png", format="png", bbox_inches="tight")
    fig_n_contacts.savefig(output_root * ".n_contacts.png", format="png", bbox_inches="tight")

    #gamma plot
    output_files=[joinpath(dirname(f), split(basename(f), ".")[1]) for f in file_model_scores]
    fig_gamma, ax_gamma, gammas, gamma_est = plot_gamma(output_files)
    fig_gamma.savefig(output_root * ".gamma.png", format="png", bbox_inches="tight")
    out_gamma = open(output_root * ".gamma.txt", "w")
    print(out_gamma, "\t")
    println(out_gamma, join(output_files, "\t"))
    println(out_gamma, "gamma inferred: \t", join(string.(gammas), "\t"))
    println(out_gamma, "gamma empirical: \t", join(string.(gamma_est), "\t"))
    close(out_gamma)



end