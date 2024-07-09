using Revise
using DiffusionEvolution, PyPlot
pygui(true)
const de = DiffusionEvolution

d = 10
n_samples = 100
T = 3

Δ=[(T+1-t)/T for t in 1:T]
figure()
begin #CREATE RANDOM DATA
    x_p = randn(d, n_samples, T)
    for t in 1:T
        x_p[:,:,t] .+= Δ[t]*(t-1.0)
        hist(x_p[1,:,t], bins=30, alpha=0.3, label="t=$t")
    end 
    legend()
    counts = Float64.(rand(1:100, n_samples, T))
    counts ./= sum(counts, dims=1)
    deltas = [1, 2, 3]
    data = collect_data(x_p, counts, deltas)
end


values,x = de.learn_gd(data, epochs=(10000,), η=(0.01,), λ=1.0, verbose=true);

begin
    figure()
    plot(values)
    xlabel("epochs")
    ylabel("log-likelihood")
end

begin
    figure()
    Σ_empirical = (data.round[end].x * data.round[end].x')./data.M
    J_inferred = de.compute_J(x, data.d)
    Λ_inferred = de.compute_lambda(J_inferred)
    Σ_inferred = de.compute_sigma(T, data, J_inferred, Λ_inferred)
    scatter(vec(Σ_empirical), vec(Σ_inferred))
end



