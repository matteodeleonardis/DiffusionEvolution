function index(i::Int, j::Int) #i<j

    return (j-1)*(j-2)÷2 + i 
end


function compute_J!(x::Pars, w::Workspace)

    for j in 1:w.d
        for i in 1:j-1
            w.J[i,j] = x[index(i,j)]
            w.J[j,i] = w.J[i,j]
        end
    end
end


function compute_inverse_J!(w::Workspace)

    w.invJ .= inv(w.J)
end


function compute_lambda!(w::Workspace)

    w.Λ .= exp(-w.J)
end


function compute_mu!(t::Int, data::Data, w::Workspace)

    w.μ .= (w.Λ^data.delta[t]) * data.round[t].x
end


function compute_sigma!(t::Int, data::Data, w::Workspace)

    w.Σ .= w.invJ*(I(w.d) - w.Λ^(2*data.delta[t]))
end


function compute_inverse_sigma!(w::Workspace)

    w.invΣ .= inv(w.Σ)
end


function compute_parameters!(x::Pars, t::Int, data::Data, w::Workspace)

    compute_J!(x, w)
    compute_inverse_J!(w)
    compute_lambda!(w)
    compute_mu!(t, data, w)
    compute_sigma!(t, data, w)
    compute_inverse_sigma!(w)

    return
end