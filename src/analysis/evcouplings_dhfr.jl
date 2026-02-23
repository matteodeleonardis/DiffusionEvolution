function run_evcouplings_analysis_dhfr(; in_nat,
    in_wt,
    in_r1,
    in_r2,
    in_r3,
    in_r4,
    in_r5,
    in_r15,
    evc_score_dir, contacts_file, output_root, file_model_scores::Vector)

    if !ispath(output_root * ".score.jld2")
        input_fasta = []
        cnt=0
        for in_f in [in_nat, in_r1, in_r2, in_r3, in_r4, in_r5, in_r15]
            seqs = read_sequences(in_f)
            for s in seqs
                cnt += 1
                push!(input_fasta, ("seq_$(cnt)", s[2]))
            end
        end
        writefasta(evc_score_dir * "all_seqs.fasta", input_fasta)
        output_ev_couplings = plmdca(evc_score_dir * "all_seqs.fasta"; min_separation=5)
        _, L = size(output_ev_couplings.htensor)
        ev_couplings_score = output_ev_couplings.score
        @save joinpath(evc_score_dir, "ev_couplings.score.jld2") ev_couplings_score L
    else
        println("Loading evcouplings scores.")
        score_load = JLD2.load(output_root * ".score.jld2")
        ev_couplings_score = score_load["ev_couplings_score"]
        L = score_load["L"]
    end

    true_contacts = npzread(contacts_file)
    ppv_ev_couplings = compute_true_positives(ev_couplings_score, true_contacts, x -> x>0.0)

    #ppv curve
    figure()
    plot(ppv_ev_couplings[1:L], label="EVCouplings")
    title("PPV curve EVCouplings")
    xticks([0, L÷2, L], ["0", "L/2", "L"])
    xlabel("number of pairs")
    ylabel("fraction correct predictions")
    legend()
    gcf().savefig(output_root * ".ppv.png", format="png", bbox_inches="tight")

    #contact plot
    figure()
    ax=gca()
    contact_plot(ev_couplings_score, true_contacts, div(L,2), ax=ax)
    ax.set_xlabel("site i")
    ax.set_ylabel("site j")
    ax.scatter([], [], color="blue", label="correct prediction")
    ax.scatter([], [], color="red", label="incorrect prediction")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    gcf().savefig(output_root * ".contact.png", format="png", bbox_inches="tight")
    
    #new contacts plot
    fig, ax, fig_n_contacts, ax_n_contacts, fig_n_new_contacts, ax_n_new_contacts, fig_acc, ax_acc = compare_new_contacts(ev_couplings_score, true_contacts, file_model_scores, div(L,2), "EVCouplings")
    fig.savefig(output_root * ".new_contacts.png", format="png", bbox_inches="tight")
    fig_n_contacts.savefig(output_root * ".n_contacts.png", format="png", bbox_inches="tight")
    fig_n_new_contacts.savefig(output_root * ".n_new_contacts.png", format="png", bbox_inches="tight")
    fig_acc.savefig(output_root * ".prediction_agreement.png", format="png", bbox_inches="tight")
    ax_acc.set_title("Agreement with EV Couplings")

    dlabel = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

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
            ax_ppv[i].plot(ppv_ev_couplings[1:L], label="ev_couplings")
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
    fig_cont, ax_cont = subplots(1, length(file_model_scores), 6, constrained_layout=true)
    for i in eachindex(file_model_scores)
        scores_d = []
        open(file_model_scores[i], "r") do io
            for line in eachline(io)
                si, sj, score = split(line)
                push!(scores_d, (parse(Int, si), parse(Int, sj), parse(Float64, score)))
            end
            contact_plot(ev_couplings_score, true_contacts, div(L,2); color_pos="purple", ax=ax_cont[i], flip=true)
            contact_plot(scores_d, true_contacts, div(L,2); color_pos="blue", ax=ax_cont[i])
            ax_cont[i].set_xlabel("site i")
            ax_cont[i].set_ylabel("site j")
            ax_cont[i].set_title("Contact Map Predictions (d=$(dlabel[i]))")

            ax_cont[i].scatter([], [], color="purple", label="positive prediction ev_couplings")
            ax_cont[i].scatter([], [], color="blue", label="postive prediction OU model")
            ax_cont[i].scatter([], [], color="red", label="incorrect prediction")
            ax_cont[i].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        end
    end
    fig_cont.savefig(output_root * ".contact_map_compare.png", format="png", bbox_inches="tight")


    #comparison with plmdca
    fig_plmdca = figure()
    ax_plmdca = gca()
    min_dist_intermediate = 12
    max_dist_intermediate = 23
    plmdca_score = JLD2.load(joinpath(dirname(output_root), "plmdca.score.jld2"))["plmdca_score"]
    contacts_plmdca = []
	predictions_plmdca = []
	for i in 1:div(L,2)
		push!(predictions_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
		if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
			push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
		end
	end

    new_contacts = []
    n_contacts = 0
    n_add_contacts = 0
    n_add_contacts_intermediate = 0
    n_add_contacts_long = 0
    n_agr_predictions = 0
    for j in 1:div(L,2)
        score_i = round(Int, ev_couplings_score[j][1])
        score_j = round(Int, ev_couplings_score[j][2])
        if (true_contacts[score_i, score_j] > 0)
            n_contacts += 1
            if !((score_i, score_j) in contacts_plmdca)
                push!(new_contacts, (score_i, score_j))
                n_add_contacts += 1
                if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                    n_add_contacts_intermediate += 1
                elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                    n_add_contacts_long += 1
                end
            end
        end
        if (score_i, score_j) in predictions_plmdca
            n_agr_predictions += 1
        end
    end
    ax_plmdca.set_title("Predicted Additional Contacts (EV Couplings vs PlmDCA)")
    ax_plmdca.set_xlabel("site i")
    ax_plmdca.set_ylabel("site j")
    ax_plmdca.matshow(true_contacts, cmap="BuGn")
    ax_plmdca.scatter(map(x->x[1], new_contacts), map(x->x[2], new_contacts), color="orangered", s=5)

    fig_plmdca.savefig(output_root * ".contacts_vs_plmdca.png", format="png", bbox_inches="tight")

    open(output_root * "_vs_plmdca.txt", "w") do io
        write(io, "Number predicted contacts: $(n_contacts)\n")
        write(io, "Number predicted additional contacts (vs plmdca): $(n_add_contacts)\n")
        write(io, "Number predicted additional contacts (intermediate range): $(n_add_contacts_intermediate)\n")
        write(io, "Number predicted additional contacts (long range): $(n_add_contacts_long)\n")
        write(io, "Agreement with plmdca: $(n_agr_predictions) out of $(div(L,2))\n")
    end

end