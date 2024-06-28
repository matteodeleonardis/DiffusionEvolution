struct Workspace

    d::Int

    μ::Matrix
    Σ::Matrix
    J::Matrix
    Λ::Matrix
end

function init_workspace(; d)
    μ = zeros(d,1)
    Σ = zeros(d,d)
    J = zeros(d,d)
    Λ = zeros(d,d)

    return Workspace(d,μ, Σ, J, Λ)
end