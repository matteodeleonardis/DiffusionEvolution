using Revise
using DiffusionEvolution, PyPlot, LinearAlgebra
pygui(true)

d = 10
nsamples =1000
T=10

λ_diag = 1.0
λ_skew = 0.1
J = λ_skew*randn(d,d) + λ_diag*I(d)
J = (J .+ J')/2
eigen(inv(J))

x = zeros(d, nsamples, T)

for s in 1:nsamples
    for t in 1:T-1
        x[:,s,t+1] = x[:,s,t] .- J*x[:,s,t] .+ randn(d)
    end
end

fig, ax = subplots(1, T, figsize=(T*4,2))
for t in 1:T
    ax[t].hist2d(x[1,:,t], x[2,:,t], bins=20)
end

counts = ones(nsamples, T) ./ nsamples
deltas = fill(1,T)
data = collect_data(x, counts, deltas)

ll_values, x_opt = de.learn_gd(data, epochs=(1000,), η=(0.01,), λ=0.0, verbose=true);
figure()
plot(ll_values)

J_inferred = de.compute_J(x_opt, data.d)
figure()
scatter(J, J_inferred)