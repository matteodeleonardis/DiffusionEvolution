using Revise
using DiffusionEvolution, PyPlot, LinearAlgebra, Statistics
const de = DiffusionEvolution
pygui(true)

d = 50
nsamples =1000
T=10

λ_diag = 1.0
λ_skew = 0.01
J = λ_skew*randn(d,d) + λ_diag*I(d)
J = (J .+ J')/2
minimum(eigen(inv(J)).values)

begin
    x = zeros(d, nsamples, T)
    for s in 1:nsamples
        for t in 1:T-1
            x[:,s,t+1] = x[:,s,t] .- J*x[:,s,t] .+ randn(d)
        end
    end
end

tpoints = 1:1:T
fig, ax = subplots(1, length(tpoints), figsize=(length(tpoints)*4,2))
for t in eachindex(tpoints)
    ax[t].hist2d(x[1,:,tpoints[t]], x[2,:,tpoints[t]], bins=20)
end

μ_eq = mean(x[:,:,end], dims=2)
figure()
plot(μ_eq, marker="o", linestyle="dashed")
Σ_eq = (x[:,:,end]*x[:,:,end]')./nsamples
figure()
scatter(vec(Σ_eq), vec(inv(J)))
cor(vec(Σ_eq), vec(inv(J)))

counts = ones(nsamples, T) ./ nsamples
deltas = fill(1,T)
data = collect_data(x, counts, deltas)

x0 = zeros(de.npars(data.d))
J_eq = inv(Σ_eq)
for i in 1:data.d
    for j in i:data.d
        x0[de.index(i,j)] = J_eq[i,j]
    end
end

ll_values, x_opt = de.learn_gd(data, x0=x0, epochs=(1000,), η=(0.01,), λ=0.0, verbose=true);
figure()
plot(ll_values)

J_inferred = de.compute_J(x_opt, data.d)
fig, ax = subplots(1,3, figsize=(4*3, 4))
ax[1].scatter(J, J_inferred)
ax[1].set_xlabel("J teacher")
ax[1].set_ylabel("J student")

ax[2].scatter(inv(J), inv(J_inferred))
ax[2].set_xlabel("inverse J teacher")
ax[2].set_ylabel("inverse J student")

ax[3].scatter(Σ_eq, inv(J_inferred))
ax[3].set_xlabel("Σ empirical")
ax[3].set_ylabel("inverse J student")