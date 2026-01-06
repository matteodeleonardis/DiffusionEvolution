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


function compare_new_contacts(plmdca_score, true_contacts, file_model_scores::Vector, pairs_threshold)

	contacts_plmdca = []
	for i in 1:pairs_threshold
		if true_contacts[plmdca_score[i][1], plmdca_score[i][2]] > 0
			push!(contacts_plmdca, (plmdca_score[i][1], plmdca_score[i][2]))
		end
	end

	n_model_scores = length(file_model_scores)
	model_scores = [readdlm(f, '\t', Float64) for f in file_model_scores]

	fig, ax = subplots(1, n_model_scores, 6)
	for i in eachindex(ax)

		model_score = model_scores[i]
		new_contacts = []
		for j in eachindex(1:pairs_threshold)
			if !(Tuple(Int.(model_score[j, 1:2])) in contacts_plmdca) && (true_contacts[Int(model_score[j, 1]), Int(model_score[j, 2])] > 0)
				push!(new_contacts, Tuple(Int.(model_score[j, 1:2])))
			end
		end
		ax[i].set_title(basename(file_model_scores[i]))
		ax[i].set_xlabel("site i")
		ax[i].set_ylabel("site j")
		ax[i].matshow(true_contacts, cmap="BuGn")
		ax[i].scatter(map(x->x[1], new_contacts), map(x->x[2], new_contacts), color="orangered", s=5)

	end

	return fig, ax
end
