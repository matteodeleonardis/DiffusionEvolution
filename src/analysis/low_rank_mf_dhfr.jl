function run_low_rank_mf_analysis_dhfr(; input_fasta, wt_fasta, contacts_file, output_root, low_rank_mf_dir, 
    file_model_scores::Vector, epsilon_rel=1.0e-8, eps_warning=1.0e-4, set_zero=false, min_dist_intermediate = 12, max_dist_intermediate = 23)

    true_contacts = npzread(contacts_file)

    output_dir = dirname(output_root)
    if ispath(joinpath(output_dir, "plmdca.score.jld2"))
        println("Loading plmdca scores.")
        score_load = JLD2.load(joinpath(output_dir, "plmdca.score.jld2"))
        plmdca_score = score_load["plmdca_score"]
        L = score_load["L"]
    else
        println("PlmDCA scores not found. Run PlmDCA before and try again...")
        return
    end

    L = length(readfasta(wt_fasta)[1][2])
    A=21
    pca = nothing
    d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

    n_model_scores = length(file_model_scores)

    n_contacts_plmdca = 0
    n_positive_contacts_ou = zeros(n_model_scores)
    n_positive_contacts_low_rank = zeros(n_model_scores)
    n_new_contacts_low_rank_vs_plmdca = zeros(n_model_scores)
    n_new_contacts_low_rank_vs_plmdca_intermediate = zeros(n_model_scores)
    n_new_contacts_low_rank_vs_plmdca_long = zeros(n_model_scores)
    n_new_contacts_ou_vs_plmdca = zeros(n_model_scores)
    n_new_contacts_ou_vs_plmdca_intermediate = zeros(n_model_scores)
    n_new_contacts_ou_vs_plmdca_long = zeros(n_model_scores)
    n_new_contacts_ou_vs_low_rank = zeros(n_model_scores)
    n_new_contacts_ou_vs_low_rank_intermediate = zeros(n_model_scores)
    n_new_contacts_ou_vs_low_rank_long = zeros(n_model_scores)

    n_agree_predictions_ou_vs_low_rank = zeros(n_model_scores)

    fig_new_contacts, ax_new_contacts = subplots(1, n_model_scores, 6; squeeze=false, constrained_layout=true)

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
        ax_ppv.plot(ppv_frobenius_zerosumgauge_apc[1:L], label="low-rank")
        title("PPV curve Low-Rank GaussDCA")
        xticks([0, L÷2, L], ["0", "L/2", "L"])
        xlabel("number of pairs")
        ylabel("fraction correct predictions")
        legend()
        fig_ppv.savefig(path_score * ".ppv.png", format="png", bbox_inches="tight")
        close(fig_ppv)

        #compare predictions
        low_rank_score = frobenius_score_zerosumgauge_apc
        contacts_low_rank = []
        contacts_plmdca = []
        new_contacts_low_rank_vs_plmdca = []
        n_add_contacts_low_rank_vs_plmdca = 0
        n_add_contacts_low_rank_vs_plmdca_intermediate = 0
        n_add_contacts_low_rank_vs_plmdca_long = 0
        new_contacts_ou_vs_plmdca = []
        predictions_low_rank = []

        for i in 1:div(L,2)
            if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
                push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
            end
        end
        if n_contacts_plmdca == 0
            n_contacts_plmdca = length(contacts_plmdca)
        end

        for i in 1:div(L,2)
            push!(predictions_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
            if true_contacts[low_rank_score[i][1], low_rank_score[i][2]] > 0
                push!(contacts_low_rank, (low_rank_score[i][1], low_rank_score[i][2]))
            end
        end

        n_add_contacts_low_rank_vs_plmdca = 0
        for c in contacts_low_rank
            if !(c in contacts_plmdca)
                n_add_contacts_low_rank_vs_plmdca += 1
                push!(new_contacts_low_rank_vs_plmdca, c)
                if min_dist_intermediate <= abs(c[1] - c[2]) <= max_dist_intermediate
                    n_add_contacts_low_rank_vs_plmdca_intermediate += 1
                elseif abs(c[1] - c[2]) >= max_dist_intermediate + 1
                    n_add_contacts_low_rank_vs_plmdca_long += 1
                end
            end
        end
        n_new_contacts_low_rank_vs_plmdca[di] = n_add_contacts_low_rank_vs_plmdca


        model_score = readdlm(file_model_scores[di], '\t', Float64)
        new_contacts_ou_vs_low_rank = []
        n_contacts_low_rank = length(contacts_low_rank)
        n_contacts_ou = 0
        n_add_contacts_ou_vs_low_rank = 0
        n_add_contacts_ou_vs_low_rank_intermediate = 0
        n_add_contacts_ou_vs_low_rank_long = 0
        n_add_contacts_ou_vs_plmdca = 0
        n_add_contacts_ou_vs_plmdca_intermediate = 0
        n_add_contacts_ou_vs_plmdca_long = 0
        n_agr_predictions_ou_vs_low_rank = 0
        for j in 1:div(L,2)
            score_i = round(Int, model_score[j, 1])
            score_j = round(Int, model_score[j, 2])
            if (true_contacts[score_i, score_j] > 0)
                n_contacts_ou += 1
                if !((score_i, score_j) in contacts_low_rank)
                    push!(new_contacts_ou_vs_low_rank, (score_i, score_j))
                    n_add_contacts_ou_vs_low_rank += 1
                    if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                        n_add_contacts_ou_vs_low_rank_intermediate += 1
                    elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                        n_add_contacts_ou_vs_low_rank_long += 1
                    end
                end
                if !((score_i, score_j) in contacts_plmdca)
                    push!(new_contacts_ou_vs_plmdca, (score_i, score_j))
                    n_add_contacts_ou_vs_plmdca += 1
                    if min_dist_intermediate <= abs(score_i - score_j) <= max_dist_intermediate
                        n_add_contacts_ou_vs_plmdca_intermediate += 1
                    elseif abs(score_i - score_j) >= max_dist_intermediate + 1
                        n_add_contacts_ou_vs_plmdca_long += 1
                    end
                end
            end
            if (score_i, score_j) in predictions_low_rank
                n_agr_predictions_ou_vs_low_rank += 1
            end
        end

        ax_new_contacts[1,di].set_title("Predicted Additional Contacts (d=$(d_values[di]))")
        ax_new_contacts[1,di].set_xlabel("site i")
        ax_new_contacts[1,di].set_ylabel("site j")
        ax_new_contacts[1,di].matshow(true_contacts, cmap="BuGn")
        ax_new_contacts[1,di].scatter(map(x->x[1], new_contacts_ou_vs_plmdca), map(x->x[2], new_contacts_ou_vs_plmdca), 
            color="orangered", s=5, label="new contacts (OU vs PlmDCA)")
        ax_new_contacts[1,di].scatter(map(x->x[2], new_contacts_low_rank_vs_plmdca), map(x->x[1], new_contacts_low_rank_vs_plmdca), 
            color="red", s=5, label="new contacts (Low-Rank vs PlmDCA)")
        ax_new_contacts[1,di].legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

        n_positive_contacts_ou[di] = n_contacts_ou
        n_positive_contacts_low_rank[di] = n_contacts_low_rank
        n_new_contacts_ou_vs_low_rank[di] = n_add_contacts_ou_vs_low_rank
        n_new_contacts_ou_vs_low_rank_intermediate[di] = n_add_contacts_ou_vs_low_rank_intermediate
        n_new_contacts_ou_vs_low_rank_long[di] = n_add_contacts_ou_vs_low_rank_long
        n_new_contacts_ou_vs_plmdca[di] = n_add_contacts_ou_vs_plmdca
        n_new_contacts_ou_vs_plmdca_intermediate[di] = n_add_contacts_ou_vs_plmdca_intermediate
        n_new_contacts_ou_vs_plmdca_long[di] = n_add_contacts_ou_vs_plmdca_long
        n_new_contacts_low_rank_vs_plmdca_intermediate[di] = n_add_contacts_low_rank_vs_plmdca_intermediate
        n_new_contacts_low_rank_vs_plmdca_long[di] = n_add_contacts_low_rank_vs_plmdca_long
        n_agree_predictions_ou_vs_low_rank[di] = n_agr_predictions_ou_vs_low_rank
    end

    fig_new_contacts.savefig(output_root * ".new_contacts_vs_plmdca.png", format="png", bbox_inches="tight")

    fig_n_contacts = figure()
    ax_n_contacts = gca()
    msize = 3
	ax_n_contacts.plot(d_values, n_positive_contacts_ou, marker="o", markersize=msize, label="positive predictions OU")
    ax_n_contacts.plot(d_values, n_positive_contacts_low_rank, marker="o", markersize=msize, label="positive predictions low-rank")
    ax_n_contacts.axhline(n_contacts_plmdca, linestyle="dashed", color="red", label="PlmDCA positive predictions")
	#ax_n_contacts.set_xticks(1:n_model_scores, [basename(f) for f in file_model_scores], rotation=90)
	ax_n_contacts.legend()
	ax_n_contacts.set_xlabel("d")
	ax_n_contacts.set_ylabel("number of contacts")
	ax_n_contacts.set_title("Predicted Additional Contacts")
    fig_n_contacts.savefig(output_root * ".n_contacts_ou_vs_low_rank.png", format="png", bbox_inches="tight")
    close(fig_n_contacts)

    fig_n_new_contacts = figure()
    ax_n_new_contacts = gca()
	ax_n_new_contacts.plot(d_values, n_new_contacts_ou_vs_low_rank, marker="o", markersize=msize, label="new contacts OU vs low-rank")
	ax_n_new_contacts.plot(d_values, n_new_contacts_ou_vs_low_rank_intermediate, marker="o", markersize=msize, label="new contacts OU vs low-rank (intermediate)")
	ax_n_new_contacts.plot(d_values, n_new_contacts_ou_vs_low_rank_long, marker="o", markersize=msize, label="new contacts OU vs low-rank (long)") 
	ax_n_new_contacts.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
	ax_n_new_contacts.set_xlabel("d")
	ax_n_new_contacts.set_ylabel("number of contacts")
	ax_n_new_contacts.set_title("Predicted Additional Contacts")
    fig_n_new_contacts.savefig(output_root * ".n_new_contacts_ou_vs_low_rank.png", format="png", bbox_inches="tight")
    close(fig_n_new_contacts)

    fig_n_new_contacts_plmdca = figure()
    ax_n_new_contacts_plmdca = gca()
	ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca, marker="o", markersize=msize, label="new contacts OU vs plmdca")
	ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca_intermediate, marker="o", markersize=msize, label="new contacts OU vs plmdca (intermediate)")
	ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_ou_vs_plmdca_long, marker="o", markersize=msize, label="new contacts OU vs plmdca (long)") 
    ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca, marker="o", markersize=msize, label="new contacts low-rank vs plmdca")
	ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca_intermediate, marker="o", markersize=msize, label="new contacts low-rank vs plmdca (intermediate)")
	ax_n_new_contacts_plmdca.plot(d_values, n_new_contacts_low_rank_vs_plmdca_long, marker="o", markersize=msize, label="new contacts low-rank vs plmdca (long)") 
	ax_n_new_contacts_plmdca.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
	ax_n_new_contacts_plmdca.set_xlabel("d")
	ax_n_new_contacts_plmdca.set_ylabel("number of contacts")
	ax_n_new_contacts_plmdca.set_title("Predicted Additional Contacts")
    fig_n_new_contacts_plmdca.savefig(output_root * ".n_new_contacts_vs_plmdca.png", format="png", bbox_inches="tight")
    close(fig_n_new_contacts_plmdca)

	fig_acc = figure()
	ax_acc = gca()
	ax_acc.plot(d_values, n_agree_predictions_ou_vs_low_rank, marker="o", markersize=msize, label="agreement with PlmDCA")
	ax_acc.axhline(div(L,2), linestyle="dashed", color="red", label="number of predictions")
	ax_acc.set_ylim(0.0, div(L,2)*1.05)
	ax_acc.legend()
	ax_acc.set_xlabel("d")
	ax_acc.set_ylabel("number of predictions")
	ax_acc.set_title("Agreement with Low-rank MF")
    fig_acc.savefig(output_root * ".agreement_low_rank.png", format="png", bbox_inches="tight")
    close(fig_acc)

    if isnothing(pca)
        pca = compute_pca([input_fasta], wt_fasta; weight=false, maxoutdim=A*L)
    end
    x_pca = apply_pca(pca, [input_fasta]; whiten=false, extreme=false, d=2)
    fig_pca = hist2d_with_marginals(x_pca[1,:], x_pca[2, :], bins=50)
    fig_pca.savefig(output_root * ".pca.png", format="png", bbox_inches="tight")
    close(fig_pca)
end