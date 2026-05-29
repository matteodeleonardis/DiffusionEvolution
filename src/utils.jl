function weighted_batch_dot(w, x, M, y)

    return dot(w, sum(x .* (M*y), dims=1))
end


function weighted_batch_dot(w, x, M)

    return weighted_batch_dot(w, x, M, x)
end


function init_cov!(x0::Pars, Xdata::Matrix{Float64}, w::Vector{Float64}; d, init_gamma = -1.0)

    #println("Initializing parameters with covariance.")
    m = mean(Xdata, Weights(w), dims=2)
    C = cov(Xdata, Weights(w), 2)
    
    @assert isapprox(C,C')
    J = svd_inv(C)
    n = sqrt(sum(abs2, J))
    J = J/n
    x_s = sqrt(J)


    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = x_s[i,j]
        end
        x0[Hindex(i, d)] = m[i]
    end  
    x0[n_index(d)] = log(n)
    
    if length(x0) == npars_gamma(d)
        x0[gamma_index(d)] = init_gamma
    end
end

function init_id!(x0::Pars; d, init_gamma = -1.0)

    #println("Initializing parameters with identity.")
    J = I(d)

    for i in 1:d
        for j in i:d
            x0[Jindex(i,j,d)] = J[i,j]
        end
        x0[Hindex(i, d)] = 0.0
    end  
    x0[n_index(d)] = 0.0
    
    if length(x0) == npars_gamma(d)
        x0[gamma_index(d)] = init_gamma
    end
end


function compute_energy(x_c::Matrix{Float64}, J::Matrix{Float64}, θ::Vector{Float64})
    
    xm = (x_c .- θ)
    return vec(sum(xm .* (J*xm), dims=1))
end

function compute_energy(x_c::Matrix{Float64}, x::Vector{Float64}, ϵ::Float64)

    d = size(x_c, 1)
    J = compute_J(x, d, ϵ)
    θ = compute_theta(x, d)

    return compute_energy(x_c, J, θ)
end


function compute_weight(x_c::Matrix{Float64}, x::Vector{Float64}, t::Int, x0::Vector{Float64},
    ϵ::Float64, λ::Float64)

    μ, Σ = compute_parameters(x, t, x0, size(x_c, 1), ϵ, λ)
    return compute_energy(x_c, inv(Σ), μ)
end


function safe_log(x::Float64, ϵ::Float64)

    return log(x + ϵ)
end


function get_potts_params(x::Pars, Wproj::Matrix{Float64}, x_mean::Vector{Float64}; d::Int, epsilon::Float64, 
    A::Int, L::Int, eps_warn = 1.0e-4, set_zero=false)

    @assert size(Wproj, 1) <= size(Wproj, 2) #projects on a smaller space
    @assert L*A == size(Wproj, 2)

    J_embedding = compute_J(x, d, epsilon)
    θ_embedding = compute_theta(x, d)
    J_potts = -(Wproj')*J_embedding*Wproj
    h_potts = vec(2 * ((Wproj*x_mean)' + θ_embedding') * J_embedding * Wproj)
    h_potts_tens = reshape(h_potts, A, L)
    J_potts_tens = permutedims(reshape(J_potts, A, L, A, L), (1,3,2,4))
    flag_warning = false

    for i in axes(J_potts_tens, 3)
        for a in 1:A
            h_potts_tens[a,i] += J_potts_tens[a,a,i,i]
            J_potts_tens[a,a,i,i] = 0.0
        end

        if set_zero
            if (!flag_warning) && (maximum(abs.(J_potts_tens[:,:,i,i])) > eps_warn)
                println("Warning: possibly large value neglected in couplings J (>= $(eps_warn))")
                flag_warning = true
            end
            J_potts_tens[:,:,i,i] .= 0.0
        end
    end
     
    if length(x)==npars_gamma(d)
        γ = get_gamma(x, d)
    else
        γ = 1.0
    end

    return (J_potts_tens, h_potts_tens, γ)
end


function svd_inv(m::Matrix{Float64})
    dec = svd(m)
    return dec.V*diagm(inv.(dec.S))*transpose(dec.U)
end


function compute_entropy(x, d, ϵ, times)

    J = compute_J(x,d,ϵ)
    s = zeros(length(times))
    for (i,t) in pairs(times)
        Λt = compute_lambda(J, x[gamma_index(d)], t)
        Σt = compute_sigma(J, Λt, d)
        s[i] = 0.5*d*log(2.0*π*exp(1.0))+0.5*logdet(Σt)
    end

    return s
end


function safe_cholesky(A, name)
    S = Symmetric(0.5 .* (A .+ A'))
    vals = eigen(S).values
    if !all(isfinite, A) || minimum(vals) <= 0
        @show name
        @show size(A)
        @show extrema(A)
        @show minimum(vals)
        @show maximum(vals)
        @show count(!isfinite, A)
        error("$name is not positive definite")
    end
    return cholesky(S)
end

#plot
function hist2d_with_marginals(x, y; bins=60, figsize=(8,8))

    fig = figure(figsize=figsize)

    # Create GridSpec layout
    gs = PyPlot.matplotlib[:gridspec][:GridSpec](
        2, 2,
        Dict(
            :width_ratios  => [4, 1],
            :height_ratios => [1, 4],
            :hspace => 0.05,
            :wspace => 0.05
        )
    )

    ax_top   = fig[:add_subplot](gs[0, 0])
    ax_main  = fig[:add_subplot](gs[1, 0], sharex=ax_top)
    ax_right = fig[:add_subplot](gs[1, 1], sharey=ax_main)

    # 2D histogram
    h = ax_main[:hist2d](x, y, bins=bins)
    fig[:colorbar](h[4], ax=ax_main)

    ax_main[:set_xlabel]("Component 1")
    ax_main[:set_ylabel]("Component 2")

    # Marginals
    ax_top[:hist](x, bins=bins)
    ax_right[:hist](y, bins=bins, orientation="horizontal")

    # Clean appearance
    ax_top[:tick_params](labelbottom=false)
    ax_right[:tick_params](labelleft=false)

    ax_top[:spines][:right][:set_visible](false)
    ax_top[:spines][:top][:set_visible](false)
    ax_right[:spines][:right][:set_visible](false)
    ax_right[:spines][:top][:set_visible](false)

    fig[:tight_layout]()

    return fig
end


function derivative_nonuniform(x, y)
    n = length(x)
    dy = similar(y)

    # Forward difference (first point)
    dy[1] = (y[2] - y[1]) / (x[2] - x[1])

    # Central differences (non-uniform grid)
    for i in 2:n-1
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]

        dy[i] =
            (-h2/(h1*(h1+h2))) * y[i-1] +
            ((h2-h1)/(h1*h2))  * y[i]   +
            (h1/(h2*(h1+h2)))  * y[i+1]
    end

    # Backward difference (last point)
    dy[n] = (y[n] - y[n-1]) / (x[n] - x[n-1])

    return dy
end
