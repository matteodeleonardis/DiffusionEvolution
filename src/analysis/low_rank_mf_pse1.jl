function run_low_rank_mf_analysis_pse1(; input_fasta, wt_fasta, contacts_file, output_root, low_rank_mf_dir, 
    file_model_scores::Vector, epsilon_rel=1.0e-8, eps_warning=1.0e-4, set_zero=false, min_dist_intermediate = 12, max_dist_intermediate = 23)

    true_contacts=JLD2.load(contacts_file)["contacts"]
    if true_contacts != true_contacts'
        true_contacts += true_contacts'
    end

    L = length(readfasta(wt_fasta)[1][2])
    A=21
    pca = nothing
    d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

    n_model_scores = length(file_model_scores)

    n_positive_contacts_ou = zeros(n_model_scores)
    n_positive_contacts_low_rank = zeros(n_model_scores)
    n_new_contacts = zeros(n_model_scores)
    n_new_contacts_intermediate = zeros(n_model_scores)
    n_new_contacts_long = zeros(n_model_scores)

    n_agree_predictions = zeros(n_model_scores)

    fig_new_contacts, ax_new_contacts = subplots(1, n_model_scores, 6; squeeze=false)

    for (di, d) in pairs(d_values)
        path_score = joinpath(low_rank_mf_dir, "low_rank_mf_$d")
        if !ispath(path_score * ".tsv")
            if isnothing(pca)
                pca = compute_pca([input_fasta], wt_fasta; weight=false, maxoutdim=A*L)
            end

            W_proj = Matrix(pca.proj[:,1:d]')
            J_low_rank = -(W_proj')*diagm(inv.(principalvars(pca)[1:d]))*W_proj
            J_low_rank_tens = permutedims(reshape(J_low_rank, A, L, A, L), (1,3,2,4))

            for i in axes(J_low_rank_tens, 3)
                for a in 1:A
                    J_low_rank_tens[a,a,i,i] = 0.0
                end

                if set_zero
                    J_low_rank_tens[:,:,i,i] .= 0.0
                end
            end

            frob_norm(x) = compute_frob_norm(x, L, 21)
            frobenius_norm_zerosumgauge = compute_norm(J_low_rank_tens, zeros(A,L), ZeroSumGauge(), frob_norm)
            frobenius_norm_zerosumgauge_apc = corr_APC(frobenius_norm_zerosumgauge)
            frobenius_score_zerosumgauge_apc = PlmDCA.compute_ranking(frobenius_norm_zerosumgauge_apc)

            open(path_score * ".tsv", "w") do io
                for (a,b,c) in frobenius_score_zerosumgauge_apc
                    println(io, a, "\t", b, "\t", c)
                end
            end
        else
            frobenius_score_zerosumgauge_apc = [(round(Int, r[1]), round(Int, r[2]), r[3]) 
                for r in eachrow(readdlm(path_score * ".tsv", '\t', Float64))]
        end

        ppv_frobenius_zerosumgauge_apc = compute_true_positives(frobenius_score_zerosumgauge_apc, true_contacts, x -> x>0.0)
        fig_ppv = figure()
        ax_ppv = gca() 
        ax_ppv.plot(ppv_frobenius_zerosumgauge_apc[1:L], label="zerosumgauge_apc")
        xticks([0, L÷2, L], ["0", "L/2", "L"])
        fig_ppv.savefig(path_score * ".ppv.png", format="png", bbox_inches="tight")
        close(fig_ppv)

        #compare predictions
        low_rank_score = frobenius_score_zerosumgauge_apc
        contacts_low_rank = []
        predictions_low_rank = []
        for i in 1:div(L,2)
            push!(predictions_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
            if true_contacts[low_rank_score[i][1], low_rank_score[i][2]] > 0
                push!(contacts_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
            end
        end

        model_score = readdlm(file_model_scores[di], '\t', Float64)
        new_contacts = []
        n_contacts_low_rank = length(contacts_low_rank)
        n_contacts_ou = 0
        n_add_contacts = 0
        n_add_contacts_intermediate = 0
        n_add_contacts_long = 0
        n_agr_predictions = 0
        for j in 1:div(L,2)
            score_i = round(Int, model_score[j, 1])
            score_j = round(Int, model_score[j, 2])
            if (true_contacts[score_i, score_j] > 0)
                n_contacts_ou += 1
                if !((score_i, score_j) in contacts_low_rank)
                    push!(new_contacts, (score_i, score_j))
                    n_add_contacts += 1
                    if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                        n_add_contacts_intermediate += 1
                    elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                        n_add_contacts_long += 1
                    end
                end
            end
            if (score_i, score_j) in predictions_low_rank
                n_agr_predictions += 1
            end
        end

        ax_new_contacts[1,di].set_title("Predicted Additional Contacts (d=$(d_values[di]))")
        ax_new_contacts[1,di].set_xlabel("site i")
        ax_new_contacts[1,di].set_ylabel("site j")
        ax_new_contacts[1,di].matshow(true_contacts, cmap="BuGn")
        ax_new_contacts[1,di].scatter(map(x->x[1], new_contacts), map(x->x[2], new_contacts), color="orangered", s=5)

        n_positive_contacts_ou[di] = n_contacts_ou
        n_positive_contacts_low_rank[di] = n_contacts_low_rank
        n_new_contacts[di] = n_add_contacts
        n_new_contacts_intermediate[di] = n_add_contacts_intermediate
        n_new_contacts_long[di] = n_add_contacts_long
        n_agree_predictions[di] = n_agr_predictions
    end

    fig_new_contacts.savefig(output_root * ".new_contacts_low_rank.png", format="png", bbox_inches="tight")

    fig_n_contacts = figure()
    ax_n_contacts = gca()
    msize = 3
	ax_n_contacts.plot(d_values, n_positive_contacts_ou, marker="o", markersize=msize, label="positive contacts (OU)")
    ax_n_contacts.plot(d_values, n_positive_contacts_low_rank, marker="o", markersize=msize, label="positive contacts (low-rank)")
	ax_n_contacts.plot(d_values, n_new_contacts, marker="o", markersize=msize, label="new contacts")
	ax_n_contacts.plot(d_values, n_new_contacts_intermediate, marker="o", markersize=msize, label="new contacts (intermediate)")
	ax_n_contacts.plot(d_values, n_new_contacts_long, marker="o", markersize=msize, label="new contacts (long)") 
	#ax_n_contacts.set_xticks(1:n_model_scores, [basename(f) for f in file_model_scores], rotation=90)
	ax_n_contacts.legend()
	ax_n_contacts.set_xlabel("d")
	ax_n_contacts.set_ylabel("number of contacts")
	ax_n_contacts.set_title("Predicted Additional Contacts")
    fig_n_contacts.savefig(output_root * ".n_contacts_low_rank.png", format="png", bbox_inches="tight")
    close(fig_n_contacts)

	fig_acc = figure()
	ax_acc = gca()
	ax_acc.plot(d_values, n_agree_predictions, marker="o", markersize=msize, label="agreement with PlmDCA")
	ax_acc.axhline(div(L,2), linestyle="dashed", color="red", label="number of predictions")
	ax_acc.set_ylim(0.0, div(L,2)*1.05)
	ax_acc.legend()
	ax_acc.set_xlabel("d")
	ax_acc.set_ylabel("number of predictions")
	ax_acc.set_title("Agreement with Low-rank MF")
    fig_acc.savefig(output_root * ".agreement_low_rank.png", format="png", bbox_inches="tight")
    close(fig_acc)
end