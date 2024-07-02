using Revise
using DiffusionEvolution, PyPlot
pygui(true)
const de = DiffusionEvolution

include("likelihood_noalloc.jl")
using DiffusionEvolution: weighted_batch_dot
using LinearAlgebra, Flux



d = 10
n_samples = 100
T = 3

begin #CREATE RANDOM DATA
    x_p = randn(d, n_samples, T)
    for t in 1:T
        x_p[:,:,t] .+= (t-1.0)
    end
    counts = Float64.(rand(1:100, n_samples, T))
    counts ./= sum(counts, dims=1)
    deltas = fill(1, T)
    data = collect_data(x_p, counts, deltas)
end

w = de.init_workspace(data)

x = randn((d^2-d)÷2)
g = zeros((d^2-d)÷2)

compute_parameters!(x, 1, data, w)

log_likelihood(x, g, data, w)
loglikelihood_noalloc(x, data, w.J, d=d)

ps = Flux.params(x)
gs = gradient(ps) do
    loglikelihood_noalloc(x, data, w.J, d=d)
end


gs[x]
g

