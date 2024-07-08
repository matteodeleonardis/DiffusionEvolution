using LinearAlgebra, Flux

using DiffusionEvolution
using DiffusionEvolution: index
const de = DiffusionEvolution

d = 3
x = randn((d^2-d)÷2)
g = zeros((d^2-d)÷2)

T=1
n_samples=1
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
 
w=de.init_workspace(d=d, n_sample=1)

begin
    de.compute_J!(x, w)
    de.compute_inverse_J!(w)
    de.compute_lambda!(w)
    de.compute_sigma!(1, data, w)
    de.compute_inverse_sigma!(w)
    de.compute_∂J!(1,2,w)
end

############################# lambda
function f1(x)
    J1 = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    #J2 = [i>j ? x[index(j,i)] : 0.0 for i in 1:d, j in 1:d]

    J=J1 + J1'
    exp(-J)
end

ps = Flux.params(x)
gs = gradient(ps) do 
    f1(x)[1,2]
end

f1(x)==w.Λ

gs[x]#[1]
(-w.∂J*w.Λ)#[1,2]


############################# lambda^2
function f2(x)
    J1 = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    #J2 = [i>j ? x[index(j,i)] : 0.0 for i in 1:d, j in 1:d]

    J=J1 + J1'
    l=exp(-2*J)
    #l^2
end

ps = Flux.params(x)
gs = gradient(ps) do 
    f2(x)[1,2]
end

f2(x)==w.Λ^2
abs.(f2(x) .- w.Λ^2)

gs[x][1]
(-2*w.∂J*w.Λ)[1,2]



############################# J^-1
function f2(x)
    J1 = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    #J2 = [i>j ? x[index(j,i)] : 0.0 for i in 1:d, j in 1:d]

    J=J1 + J1'
    inv(J)
end

ps = Flux.params(x)
gs = gradient(ps) do 
    f2(x)[1,2]
end

f2(x)==w.invJ

gs[x]
-w.invJ*w.∂J*w.invJ


############################# (1-lambda^2)
function f3(x)
    J1 = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    #J2 = [i>j ? x[index(j,i)] : 0.0 for i in 1:d, j in 1:d]

    J=J1 + J1'
    l=exp(-J)
    I(d)-l^2
end

ps = Flux.params(x)
gs = gradient(ps) do 
    f3(x)[1,2]
end

f3(x) == w.J*w.Σ
abs.(f3(x) .- w.J*w.Σ)

gs[x]
2*w.∂J*w.Λ


############################################# sigma
function f3(x)
    J1 = [i<j ? x[index(i,j)] : 0.0 for i in 1:d, j in 1:d]
    J=J1 + J1'
    l=exp(-J)
    inv(J)*(I(d)-l^2)
end

ps = Flux.params(x)
gs = gradient(ps) do 
    f3(x)[1,2]
end

f3(x) == w.Σ
gs[x]
-w.invJ*w.∂J*w.Σ + 2w.invJ*w.∂J*w.Λ

