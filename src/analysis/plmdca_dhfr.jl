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
    fig, ax = compare_new_contacts(plmdca_score, true_contacts, file_model_scores, L)
    fig.savefig(output_root * ".new_contacts.png", format="png", bbox_inches="tight")

end