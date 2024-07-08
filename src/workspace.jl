struct Workspace

    d::Int

    μ::Matrix{Float64}
    Σ::Matrix{Float64}
    J::Matrix{Float64}
    Λ::Matrix{Float64}
    invΣ::Matrix{Float64}
    invJ::Matrix{Float64}
    ∂J::Matrix{Float64}
end

function init_workspace(; d, n_sample)
    μ = zeros(d, n_sample)
    Σ = zeros(d,d)
    J = zeros(d,d)
    Λ = zeros(d,d)
    invΣ = zeros(d,d)
    invJ = zeros(d,d)
    ∂J = zeros(d,d)

    return Workspace(d, μ, Σ, J, Λ, invΣ, invJ, ∂J)
end


function init_workspace(data::Data)

    init_workspace(d=size(data.round[1].x, 1), n_sample=size(data.round[1].x, 2))
end


function compute_∂J!(i::Int, j::Int, w::Workspace)
    w.∂J .= 0.0
    if i == j
        w.∂J[i,i] = 1.0
    else
        w.∂J[i,j] = 1.0
        w.∂J[j,i] = 1.0
    end
end
