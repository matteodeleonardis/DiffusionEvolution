function random_pars(λ_diag, λ_skew, θ0, θ1; d)

    J = λ_skew*randn(d,d) + λ_diag*I(d)
    J = (J .+ J')/2
    θ = θ0 .+ (θ1 .* randn(d))
    min_eigv = minimum(eigen(inv(J)).values)

    return (J, θ, min_eigv)
end

function simulate_ou_process(J, θ; nsamples, T)
    d = length(θ)
    x = zeros(d, nsamples, T)
    for s in 1:nsamples
        for t in 1:T-1
            x[:,s,t+1] = x[:,s,t] .- J*(x[:,s,t] .- θ) .+ randn(d)
        end
    end

    return x
end