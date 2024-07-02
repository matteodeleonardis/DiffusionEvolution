struct Workspace

    d::Int

    μ::Matrix{Float64}
    Σ::Matrix{Float64}
    J::Matrix{Float64}
    Λ::Matrix{Float64}
    invΣ::Matrix{Float64}
    invJ::Matrix{Float64}
end

function init_workspace(; d, n_sample)
    μ = zeros(d, n_sample)
    Σ = zeros(d,d)
    J = zeros(d,d)
    Λ = zeros(d,d)
    invΣ = zeros(d,d)
    invJ = zeros(d,d)

    return Workspace(d, μ, Σ, J, Λ, invΣ, invJ)
end


function init_workspace(data::Data)

    init_workspace(d=size(data.round[1].x, 1), n_sample=size(data.round[1].x, 2))
end