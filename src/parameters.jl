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


function compute_lambda!(w::Workspace)

    w.Λ .= exp(-w.J)
end


function compute_sigma!(w::Workspace)

    w.Σ .= inv(w.J)*(I(w.d) - w.Λ^2)
end
