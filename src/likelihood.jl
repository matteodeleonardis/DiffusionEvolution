function log_likelihood(x::Pars, g::Pars, data::Data, w::Workspace)

    ll = 0.0
    
    for t in eachindex(data.round)
        compute_parameters!(x, t, data, w)
        ll += log(det(w.Σ)) #+ weighted_batch_dot(data.round[t].w, (data.round[t].x .- w.μ), w.invΣ)
    end

    #computing gradient

    for i in 1:w.d
        for j in i+1:w.d
            compute_∂J!(i,j,w)
            contrib1 = 0.0
            contrib2 = 0.0
            contrib3 = 0.0
            for t in eachindex(data.round)
                compute_parameters!(x, t, data, w)
                contrib1 -= tr(w.invΣ*w.invJ*w.∂J*(w.Σ-2*data.delta[t]*w.Λ))
                #=contrib2 += 4*data.delta[t]*w.Λ[i,j]*
                    weighted_batch_dot(data.round[t].w, (data.round[t].x .- w.μ), w.invΣ, data.round[t].x)
                contrib3 -= 2*w.invJ[i,j]*weighted_batch_dot(data.round[t].w, (data.round[t].x .- w.μ), 
                    (I(w.d) - (2*data.delta[t]*w.Λ)*w.invΣ))=#

            end
            g[index(i,j)] =  contrib1 #+contrib2 + contrib3
        end
    end



    return ll
end