function corr_APC(S)
	N = size(S, 1)
    Si = sum(S, dims=1)
    Sj = sum(S, dims=2)
    Sa = sum(S) * (1 - 1/N)

    S -= (Sj * Si) / Sa
    return S
end


function compute_frob_norm(J::Array{Float64,4},N,q)
    FN = fill(0.0, N,N)
    for i=1:N-1
        for j=i+1:N
            FN[i,j] = norm(J[1:q-1,1:q-1,i,j],2)
            FN[j,i] =FN[i,j]
        end
    end
    return FN
end


function wt_norm(J, wt)
	L = length(wt)
	S = zeros(L,L)
	for i in 1:L
		for j in i+1:L
			S[i,j] = sqrt(0.5*( sum(x->x^2, J[:, wt[j], i, j]) + sum(x->x^2, J[wt[i], :, i, j]) ))
			S[j,i] = S[i,j]
		end
	end

	return S
end


function compute_norm(J, h, this_gauge, mat_norm)
	
	A,_,L,_=size(J)
    if isnothing(this_gauge) 
        norm2 = mat_norm(J)
        return norm2
    else
    	J_gauged, h_gauged = gauge(J, h, this_gauge)
    	norm2 = mat_norm(J_gauged)
    	return norm2
    end
end

function compute_true_positives(rank, contacts, is_contact)

	n_pairs = length(rank)
	tp = zeros(n_pairs)
	tp[1] = Float64(is_contact(contacts[rank[1][1], rank[1][2]]))
	for k in 2:n_pairs
    
    	tp[k] = ( tp[k-1]*(k-1) + Float64(is_contact(contacts[rank[k][1], rank[k][2]])) )/k
	end
    
    return tp
end


function contact_plot(rank, contact, n_contacts; cmap="BuGn", color_pos="blue", color_neg="red", s=5, ax=nothing, flip=false)

	if isnothing(ax)
		matshow(contact, cmap=cmap)
	else
		ax.matshow(contact, cmap=cmap)
	end
	
	if !flip #plots contacts in the lower triangle of the contact map
		for r in rank[1:n_contacts]
			if contact[r[1], r[2]] > 0
				if isnothing(ax)    
					scatter([r[1]], [r[2]], color=color_pos, s=s)
				else
					ax.scatter([r[1]], [r[2]], color=color_pos, s=s)
				end
			else
				if isnothing(ax)
					scatter([r[1]], [r[2]], color=color_neg, s=s)
				else
					ax.scatter([r[1]], [r[2]], color=color_neg, s=s)
				end
			end
		end
	else #plots contacts in the upper triangle of the contact map
		for r in rank[1:n_contacts]
			if contact[r[1], r[2]] > 0
				if isnothing(ax)    
					scatter([r[2]], [r[1]], color=color_pos, s=s)
				else
					ax.scatter([r[2]], [r[1]], color=color_pos, s=s)
				end
			else
				if isnothing(ax)
					scatter([r[2]], [r[1]], color=color_neg, s=s)
				else
					ax.scatter([r[2]], [r[1]], color=color_neg, s=s)
				end
			end
		end
	end
end


function compare_new_contacts(plmdca_score, true_contacts, file_model_scores::Vector, pairs_threshold, label1, label2; 
	min_dist_intermediate = 12, max_dist_intermediate = 23)

	contacts_plmdca = []
	predictions_plmdca = []
	for i in 1:pairs_threshold
		push!(predictions_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
		if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
			push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
		end
	end

	n_model_scores = length(file_model_scores)
	model_scores = [readdlm(f, '\t', Float64) for f in file_model_scores]

	n_positive_contacts = zeros(n_model_scores)
	n_new_contacts = zeros(n_model_scores)
	n_new_contacts_intermediate = zeros(n_model_scores)
	n_new_contacts_long = zeros(n_model_scores)
	n_agree_predictions = zeros(n_model_scores)

	println("n_model_scores: ", n_model_scores)

	d_label = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

	fig, ax = subplots(1, n_model_scores, 6; squeeze=false, constrained_layout=true)

	for i in 1:n_model_scores

		model_score = model_scores[i]
		new_contacts = []
		n_contacts = 0
		n_add_contacts = 0
		n_add_contacts_intermediate = 0
		n_add_contacts_long = 0
		n_agr_predictions = 0
		for j in 1:pairs_threshold
			score_i = round(Int, model_score[j, 1])
			score_j = round(Int, model_score[j, 2])
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
		ax[1,i].set_title("Predicted Additional Contacts $label1 vs $label2 (d=$(d_label[i]))")
		ax[1,i].set_xlabel("site i")
		ax[1,i].set_ylabel("site j")
		ax[1,i].matshow(true_contacts, cmap="BuGn")
		ax[1,i].scatter(map(x->x[1], new_contacts), map(x->x[2], new_contacts), color="orangered", s=5)

		n_positive_contacts[i] = n_contacts
		n_new_contacts[i] = n_add_contacts
		n_new_contacts_intermediate[i] = n_add_contacts_intermediate
		n_new_contacts_long[i] = n_add_contacts_long
		n_agree_predictions[i] = n_agr_predictions

	end

	fig_n_contacts = figure()
	ax_n_contacts = gca()
	msize = 3
	ax_n_contacts.plot(d_label, n_positive_contacts, marker="o", markersize=msize, label="positive contacts $label1")
	ax_n_contacts.axhline(length(contacts_plmdca), linestyle="dashed", color="red", label=label2)
	#ax_n_contacts.set_xticks(1:n_model_scores, [basename(f) for f in file_model_scores], rotation=90)
	ax_n_contacts.set_ylim(0.0, length(contacts_plmdca)*1.05)
	ax_n_contacts.legend()
	ax_n_contacts.set_xlabel("d")
	ax_n_contacts.set_ylabel("number of contacts")
	ax_n_contacts.set_title("Correctly Predicted Contacts")

	fig_n_new_contacts = figure()
	ax_n_new_contacts = gca()
	ax_n_new_contacts.plot(d_label, n_new_contacts, marker="o", markersize=msize, label="new contacts $label1 vs $label2")
	ax_n_new_contacts.plot(d_label, n_new_contacts_intermediate, marker="o", markersize=msize, label="intermediate-range")
	ax_n_new_contacts.plot(d_label, n_new_contacts_long, marker="o", markersize=msize, label="long-range") 
	ax_n_new_contacts.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
	ax_n_new_contacts.set_xlabel("d")
	ax_n_new_contacts.set_ylabel("number of contacts")
	ax_n_new_contacts.set_title("Correctly Predicted Additional Contacts $label1 vs $label2")



	fig_acc = figure()
	ax_acc = gca()
	ax_acc.plot(d_label, n_agree_predictions, marker="o", markersize=msize, label="agreement")
	ax_acc.axhline(pairs_threshold, linestyle="dashed", color="red", label="number of predictions")
	ax_acc.set_ylim(0.0, pairs_threshold*1.05)
	ax_acc.legend()
	ax_acc.set_xlabel("d")
	ax_acc.set_ylabel("number of common predictions")
	ax_acc.set_title("Agreement $label1 with $label2")


	return fig, ax, fig_n_contacts, ax_n_contacts, fig_n_new_contacts, ax_n_new_contacts, fig_acc, ax_acc 
end

function compare_new_contacts_combined(plmdca_score, true_contacts, combined_msa_score, pairs_threshold; 
	min_dist_intermediate = 12, max_dist_intermediate = 23)

	contacts_plmdca = []
	additional_contacts = []
	additional_contacts_intermediate = []
	additional_contacts_long = []
	for i in 1:pairs_threshold
		plmdca_i, plmdca_j = plmdca_score[i][1], plmdca_score[i][2]
		if true_contacts[plmdca_i, plmdca_j] > 0
			push!(contacts_plmdca, (plmdca_i, plmdca_j))
		end
	end

	for i in 1:pairs_threshold
		combined_msa_i, combined_msa_j = combined_msa_score[i][1], combined_msa_score[i][2]
		if (true_contacts[combined_msa_i, combined_msa_j] > 0) && !((combined_msa_i, combined_msa_j) in contacts_plmdca)
			push!(additional_contacts, (combined_msa_i, combined_msa_j))
			if min_dist_intermediate <= abs(combined_msa_i - combined_msa_j) <= max_dist_intermediate
				push!(additional_contacts_intermediate, (combined_msa_i, combined_msa_j))
			elseif abs(combined_msa_i - combined_msa_j) >= max_dist_intermediate + 1
				push!(additional_contacts_long, (combined_msa_i, combined_msa_j))
			end
		end
	end

	


	return additional_contacts, additional_contacts_intermediate, additional_contacts_long 
end
