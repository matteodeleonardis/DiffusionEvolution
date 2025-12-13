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
