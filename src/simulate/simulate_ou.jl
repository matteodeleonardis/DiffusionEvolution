function random_pars(λ_diag, λ_skew, θ0, θ1; d)

    J = λ_skew*randn(d,d) + λ_diag*I(d)
    J = (J .+ J')/2
    θ = θ0 .+ (θ1 .* randn(d))
    eigen_invJ = eigen(inv(J))
    min_eigv = minimum(eigen_invJ.values)
    max_eigv = maximum(eigen_invJ.values)

    return (J, θ, min_eigv, max_eigv)
end

function simulate_ou_process(J, θ, γ; nsamples, T, err_sym=0.0)
    d = length(θ)
    x = zeros(d, nsamples, T)

    Λ = compute_lambda(J, γ, d) #propagator Λ for Δt=1
    Σ = compute_sigma(J, Λ, d)
    err = maximum(abs.(Σ .- Σ'))
    if err > err_sym
        println("Σ has an high error")
        Σ .+= Σ'
        Σ .*= 0.5
    elseif err > 0.0
        println("Σ has been symmetrized")
        Σ .+= Σ'
        Σ .*= 0.5
    else
        println("Σ is symmetric")
    end
    @assert issymmetric(Σ)

    for s in 1:nsamples    
        for t in 2:T
            μ = compute_mu(x[:,s,t-1], Λ, θ, d)
            g = MvNormal(μ, Σ)
            x[:,s,t] .= rand(g)
        end
    end

    return x
end