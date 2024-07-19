using Revise
using DiffusionEvolution, PyPlot, LinearAlgebra, Statistics
const de = DiffusionEvolution
pygui(true)

d = 5
nsamples =1000
T=5

begin #interesting values are λ_diag = 1.0, 0.001 and λ_skew = 0.1, 0.001 (modify the learning rate for convergence)
    λ_diag = 1.0
    λ_skew = 0.3
    J = λ_skew*randn(d,d) + λ_diag*I(d)
    J = (J .+ J')/2
    θ = 1.0 .+ randn(d)
    minimum(eigen(inv(J)).values)
end

x = simulate_ou_process(J, θ, nsamples=nsamples, T=T)

begin
    tpoints = 1:1:T
    fig, ax = subplots(1, length(tpoints), figsize=(length(tpoints)*4,2))
    for t in eachindex(tpoints)
        ax[t].hist2d(x[1,:,tpoints[t]], x[2,:,tpoints[t]], bins=50)
        ax[t].scatter(θ[1], θ[2], marker="o", color="red")
    end
end

begin
    μ_eq = mean(x[:,:,end], dims=2)
    figure()
    scatter(θ, μ_eq)
    xlabel("teacher θ")
    ylabel("empirical average (last round)")

    Σ_eq = ((x[:,:,end] .- μ_eq)*(x[:,:,end] .- μ_eq)')./nsamples
    figure()
    scatter(vec(Σ_eq), vec(inv(J)))
    cor(vec(Σ_eq), vec(inv(J)))
    xlabel("teacher Σ")
    ylabel("empirical covariance (last round)")

end

counts = ones(nsamples, T) ./ nsamples
deltas = fill(1,T-1)
data = collect_data(x, counts, deltas)

de.log_likelihood(x_opt, data, 0.0, 0.0)

min_ll, x_opt, status = learn_nlopt(data, initialize=T, λ=0.0, prior=0.01, ftol_rel=1e-9)

begin 
    J_inferred = de.compute_J(x_opt, data.d)
    fig, ax = subplots(1,4, figsize=(6*4, 4))
    ax[1].scatter(J, J_inferred)
    ax[1].set_xlabel("J teacher")
    ax[1].set_ylabel("J student")

    ax[2].scatter(inv(J), inv(J_inferred))
    ax[2].set_xlabel("inverse J teacher")
    ax[2].set_ylabel("inverse J student")

    ax[3].scatter(Σ_eq, inv(J_inferred))
    ax[3].set_xlabel("Σ empirical")
    ax[3].set_ylabel("inverse J student")

    ax[4].scatter(θ, x_opt[[de.Hindex(i,data.d) for i in 1:data.d]])
    ax[4].set_xlabel("θ")
    ax[4].set_ylabel("inferred θ")
end