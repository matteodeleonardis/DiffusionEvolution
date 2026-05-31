function plot_distribution(x_opt, data, times, output_root; lambda, epsilon_J, epsilon_sigma)
    PyPlot.matplotlib.rcParams["svg.fonttype"] = "none"
    fig_emp_dist, ax_emp_dist = subplots(1, length(times), 6)
    if data.d > 1
        for i in eachindex(times)
            ax_emp_dist[i].hist2d(data.round[i].x[1,:], data.round[i].x[2,:], weights=data.round[i].w, bins=50)
            ax_emp_dist[i].scatter([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
            ax_emp_dist[i].set_title("t=$(times[i])")
            ax_emp_dist[i].set_xlabel("PC1")
            ax_emp_dist[i].set_ylabel("PC2")
        end
    else
        for i in eachindex(times)
            ax_emp_dist[i].hist(data.round[i].x[1,:], weights=data.round[i].w, bins=50)
            ax_emp_dist[i].axvline(data.x0[1], color="red")
            ax_emp_dist[i].set_title("t=$(times[i])")
            ax_emp_dist[i].set_xlabel("PC1")
            ax_emp_dist[i].set_ylabel("pdf")
        end
    end
    fig_emp_dist.savefig(output_root * ".emp_dist.png", format="png", bbox_inches="tight")
    xlim = ax_emp_dist[1].get_xlim()
    ylim = ax_emp_dist[1].get_ylim()

    mu, sigma, eq_theta, eq_sigma = infer_series(x_opt, data, times; lambda=lambda, epsilon_J=epsilon_J, epsilon_sigma=epsilon_sigma)
    fig_inf_dist, ax_inf_dist = subplots(1, length(times), 6)
    if data.d > 1
        for i in eachindex(times)
            samples = rand(MultivariateNormal(mu[times[i]], sigma[times[i]]), 1000)
            ax_inf_dist[i].hist2d(samples[1,:], samples[2,:], bins=50)
            ax_inf_dist[i].scatter([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
            ax_inf_dist[i].set_title("t=$(times[i])")
            ax_inf_dist[i].set_xlabel("PC1")
            ax_inf_dist[i].set_ylabel("PC2")
            ax_inf_dist[i].set_xlim(xlim)
            ax_inf_dist[i].set_ylim(ylim)
        end
    else
        for i in eachindex(times)
            samples = rand(Normal(mu[times[i]][1], sigma[times[i]][1,1]), 1000)
            ax_inf_dist[i].hist(samples, bins=50)
            ax_inf_dist[i].axvline(data.x0[1], color="red")
            ax_inf_dist[i].set_title("t=$(times[i])")
            ax_inf_dist[i].set_xlabel("PC1")
            ax_inf_dist[i].set_ylabel("pdf")
            ax_inf_dist[i].set_xlim(xlim)
        end
    end
    fig_inf_dist.savefig(output_root * ".inf_dist.png", format="png", bbox_inches="tight")

    figure()
    fig = gcf()
    ax = gca()
    if data.d > 1
        samples = rand(MultivariateNormal(eq_theta, eq_sigma), 1000)
        ax.hist2d(samples[1,:], samples[2,:], bins=50)
        ax.scatter([data.x0[1]], [data.x0[2]], marker="o", color="red", s=30)
        ax.set_title("t=+∞")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else
        samples = rand(Normal(eq_theta[1], eq_sigma[1,1]), 1000)
        ax.hist(samples, bins=50)
        ax.axvline(data.x0[1], color="red")
        ax.set_title("t=+∞")
        ax.set_xlabel("PC1")
        ax.set_ylabel("pdf")
    end
    fig.savefig(output_root * ".inf_dist_equilibrium.png", format="png", bbox_inches="tight")
end



function compute_scores(J_tens, h_tens, wt, L, output_root)
    frob_norm(x) = compute_frob_norm(x, L, 21)
    frobenius_norm_zerosumgauge = compute_norm(J_tens, h_tens, ZeroSumGauge(), frob_norm)
    frobenius_norm_zerosumgauge_apc = corr_APC(frobenius_norm_zerosumgauge)
    frobenius_norm_wildtypegauge = compute_norm(J_tens, h_tens, WildType(wt), frob_norm)
    frobenius_norm_wildtypegauge_apc = corr_APC(frobenius_norm_wildtypegauge)

    frobenius_score_zerosumgauge = PlmDCA.compute_ranking(frobenius_norm_zerosumgauge)
    frobenius_score_zerosumgauge_apc = PlmDCA.compute_ranking(frobenius_norm_zerosumgauge_apc)
    frobenius_score_wildtypegauge = PlmDCA.compute_ranking(frobenius_norm_wildtypegauge)
    frobenius_score_wildtypegauge_apc = PlmDCA.compute_ranking(frobenius_norm_wildtypegauge_apc)

    open(output_root * ".scores.zerosumgauge.tsv", "w") do io
        for (a,b,c) in frobenius_score_zerosumgauge
            println(io, a, "\t", b, "\t", c)
        end
    end

    open(output_root * ".scores.zerosumgauge_apc.tsv", "w") do io
        for (a,b,c) in frobenius_score_zerosumgauge_apc
            println(io, a, "\t", b, "\t", c)
        end
    end

    open(output_root * ".scores.wildtypegauge.tsv", "w") do io
        for (a,b,c) in frobenius_score_wildtypegauge
            println(io, a, "\t", b, "\t", c)
        end
    end

    open(output_root * ".scores.wildtypegauge_apc.tsv", "w") do io
        for (a,b,c) in frobenius_score_wildtypegauge_apc
            println(io, a, "\t", b, "\t", c)
        end
    end

    return frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc
end


function compute_ppv(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
    frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)
    PyPlot.matplotlib.rcParams["svg.fonttype"] = "none"

    ppv_frobenius_zerosumgauge = compute_true_positives(frobenius_score_zerosumgauge, true_contacts, x -> x>0.0) 
    ppv_frobenius_zerosumgauge_apc = compute_true_positives(frobenius_score_zerosumgauge_apc, true_contacts, x -> x>0.0) 
    ppv_frobenius_wildtypegauge = compute_true_positives(frobenius_score_wildtypegauge, true_contacts, x -> x>0.0) 
    ppv_frobenius_wildtypegauge_apc = compute_true_positives(frobenius_score_wildtypegauge_apc, true_contacts, x -> x>0.0)

    figure()
    plot(ppv_frobenius_zerosumgauge[1:L], label="zerosumgauge")
    plot(ppv_frobenius_zerosumgauge_apc[1:L], label="zerosumgauge_apc")
    plot(ppv_frobenius_wildtypegauge[1:L], label="wildtypegauge")
    plot(ppv_frobenius_wildtypegauge_apc[1:L], label="wildtypegauge_apc")
    xticks([0, L÷2, L], ["0", "L/2", "L"])
    legend()
    gcf().savefig(output_root * ".ppv.png", format="png", bbox_inches="tight")

    return ppv_frobenius_zerosumgauge, ppv_frobenius_zerosumgauge_apc,
        ppv_frobenius_wildtypegauge, ppv_frobenius_wildtypegauge_apc
end


function print_contact_plot(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
    frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)
    PyPlot.matplotlib.rcParams["svg.fonttype"] = "none"

    fig_contact, ax_contact = subplots(1, 4, 6)
    contact_plot(frobenius_score_zerosumgauge, true_contacts, L, ax=ax_contact[1])
    ax_contact[1].set_title("Contacts ZeroSum Gauge")
    contact_plot(frobenius_score_zerosumgauge_apc, true_contacts, L, ax=ax_contact[2])
    ax_contact[2].set_title("Contacts ZeroSum Gauge APC")
    contact_plot(frobenius_score_wildtypegauge, true_contacts, L, ax=ax_contact[3])
    ax_contact[3].set_title("Contacts WildType Gauge")
    contact_plot(frobenius_score_wildtypegauge_apc, true_contacts, L, ax=ax_contact[4])
    ax_contact[4].set_title("Contacts WildType Gauge APC")

    map(x -> x.set_xlabel("site i"), ax_contact)
    map(x -> x.set_ylabel("site j"), ax_contact)

    fig_contact.savefig(output_root * ".contact.png", format="png", bbox_inches="tight")
end

function plot_gamma(output::Vector)

    gammas = zeros(length(output))
    gamma_est = zeros(length(output))
    for i in eachindex(output)
        pars_file = output[i] * ".pars.jld2"
        settings_file = output[i] * ".settings.jld2"
        x = JLD2.load(pars_file)["x_opt"]
        gamma_min = JLD2.load(pars_file)["gamma_min"]
        d = JLD2.load(settings_file)["model_settings"].d
        gammas[i] = get_gamma(x, d)
        gamma_est[i] = gamma_min
    end


    figure()
    fig = gcf()
    ax = gca()

    ax.plot(eachindex(gammas), gammas, label="inferred")
    ax.plot(eachindex(gammas), gamma_est, label="empirical")
    ax.legend()
    ax.set_xticks(eachindex(gammas), [basename(o) for o in output], rotation=90)

    return fig, ax, gammas, gamma_est
end


function compute_log_likelihood_variants(x::Pars,  data::Data, λ::Float64, ϵ_J::Float64, ϵ_Σ::Float64, output_root::String)

    fig, ax = subplots(3, length(data.time), 6)    

    for i in eachindex(data.time)
        ll_vars = log_likelihood_variants(x, data, i, λ, ϵ_J, ϵ_Σ)
        log_counts = log.(data.round[i].w .+ 1e-12)
        idx = log_counts .> log(1e-8)
        rho_log = cor(ll_vars[idx], log_counts[idx])
        ax[1,i].scatter(ll_vars[idx], log_counts[idx]   , alpha=0.5)
        ax[1,i].set_title("t=$(data.time[i]), ρ=$(round(rho_log, digits=3))")
        ax[1,i].set_xlabel("log-likelihood variants")
        ax[1,i].set_ylabel("log (normalized) counts")

        rho = cor(exp.(ll_vars[idx]), data.round[i].w[idx])
        ax[2,i].scatter(exp.(ll_vars[idx]), data.round[i].w[idx], alpha=0.5)
        ax[2,i].set_title("t=$(data.time[i]), ρ=$(round(rho, digits=3)), all points")
        ax[2,i].set_xlabel("likelihood variants")
        ax[2,i].set_ylabel("(normalized) counts")

        observed = data.round[i].w .> 1e-8
        ax[3,i].hist(ll_vars[observed], alpha=0.5, label="observed")
        ax[3,i].hist(ll_vars[.!observed], alpha=0.5, label="floor")
        ax[3,i].set_title("t=$(data.time[i])")
        ax[3,i].set_xlabel("log-likelihood variants")
        ax[3,i].set_ylabel("count")
        ax[3,i].legend()
    end

    fig.savefig(output_root * ".likelihood_vs_counts.png", format="png", bbox_inches="tight")
end

function compute_mean_covariance(x::Pars,  
    data::Data, lambda::Float64, 
    epsilon_J::Float64, 
    epsilon_sigma::Float64, 
    output_root::String)

    fig, ax = subplots(2, length(data.time), 6)

    J = compute_J(x, data.d, epsilon_J)
    theta = compute_theta(x, data.d)
    gamma = get_gamma(x, data.d)
    for t in eachindex(data.round)
        mu, sigma = compute_parameters(J, theta, gamma, data.time[t], data.x0, data.d, lambda, epsilon_sigma)
        mu_emp = dropdims(mean(data.round[t].x[1:data.d, :], Weights(data.round[t].w), dims=2), dims=2)
        sigma_emp = cov(data.round[t].x[1:data.d, :], Weights(data.round[t].w), 2)
        rho_mu = cor(mu, mu_emp)
        rho_sigma = cor(vec(sigma), vec(sigma_emp))

        ax[1,t].scatter(mu, mu_emp)
        ax[1,t].set_title("mean t=$(data.time[t]), ρ=$(round(rho_mu, digits=3))")
        ax[1,t].set_xlabel("inferred mean")
        ax[1,t].set_ylabel("empirical mean")

        ax[2,t].scatter(vec(sigma), vec(sigma_emp))
        ax[2,t].set_title("covariance t=$(data.time[t]), ρ=$(round(rho_sigma, digits=3))")
        ax[2,t].set_xlabel("inferred covariance")
        ax[2,t].set_ylabel("empirical covariance")
    end

    fig.savefig(output_root * ".mean_covariance.png", format="png", bbox_inches="tight")
end

