using Revise
using DiffusionEvolution, PyPlot, LinearAlgebra, Statistics
const de = DiffusionEvolution
pygui(true)

d = 20
nsamples =1000
T=10

begin #interesting values are λ_diag = 1.0, 0.001 and λ_skew = 0.1, 0.001 (modify the learning rate for convergence)
    λ_diag = 1.0
    λ_skew = 0.1
    J = λ_skew*randn(d,d) + λ_diag*I(d)
    J = (J .+ J')/2
    θ = 10.0 .+ randn(d)
    minimum(eigen(inv(J)).values)
end

begin
    x = zeros(d, nsamples, T)
    for s in 1:nsamples
        for t in 1:T-1
            x[:,s,t+1] = x[:,s,t] .- J*(x[:,s,t] .- θ) .+ randn(d)
        end
    end
end

begin
    tpoints = 1:2:T
    fig, ax = subplots(1, length(tpoints), figsize=(length(tpoints)*4,2))
    for t in eachindex(tpoints)
        ax[t].hist2d(x[1,:,tpoints[t]], x[2,:,tpoints[t]], bins=50)
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
deltas = fill(1,T)
data = collect_data(x, counts, deltas)

ll_values, x_opt = de.learn_gd(data, initialize=T, epochs=(1000,), η=(0.001,), λ=0.0, verbose=true);

x0 = zeros(de.npars(d))
de.init_cov!(x0, data.round[T].x, data.round[T].w, d=d)
μ0, Σ0 = de.compute_parameters(x0, T, data, d)
Σ0
log(det(Σ0))
x0[end-20:end] .= 0.0
de.log_likelihood(x0, data, 0.0)

figure()
plot(ll_values)

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