function reprint_inference_dhfr(; output_root, file_model_scores)
    PyPlot.matplotlib.rcParams["svg.fonttype"] = "none"

    
    setting_path=joinpath(dirname(file_model_scores[1]), split(basename(file_model_scores[1]), ".")[1]*".settings.jld2")
    settings = JLD2.load(setting_path)["model_settings"]
    whiten = settings.whiten
    weight = settings.weight
    extreme = settings.extreme
    lambda = settings.lambda
    epsilon_J = settings.epsilon_J
    epsilon_sigma = settings.epsilon_sigma
    d_values = [parse(Int, split(split(basename(f), ".")[1], "_")[4]) for f in file_model_scores]

    data_dir = joinpath(@__DIR__, "../../data")

    # experiment data
    file0 = joinpath(data_dir, "dhfr/mDHFR.fasta")
    file1 = joinpath(data_dir, "dhfr/Round1_Q15_C10_aa.aln")
    file2 = joinpath(data_dir, "dhfr/Round2_Q15_C10_aa.aln")
    file3 = joinpath(data_dir, "dhfr/Round3_Q15_C10_aa.aln")
    file4 = joinpath(data_dir, "dhfr/Round4_Q15_C10_aa.aln")
    file5 = joinpath(data_dir, "dhfr/Round5_Q15_C10_aa.aln")
    file15 = joinpath(data_dir, "dhfr/Gen15_aa.aln")

    times = [1,2,3,4,5, 15]


    # natural sequences
    file_nat = joinpath(data_dir, "dhfr/mDHFR_clean.fasta")

    data, pca = process_data(file_nat, file0, [file1, file2, file3, file4, file5, file15], times; whiten=whiten, weight=weight,
                            d=maximum(d_values), extreme=extreme)


    for (di, d) in pairs(d_values)
        path_pars = joinpath(dirname(file_model_scores[di]), split(basename(file_model_scores[di]), ".")[1]*".pars.jld2")
        x_pars = JLD2.load(path_pars)["x_opt"]
        output_root_d = joinpath(output_root * string(d), basename(output_root) * "$(d)")
        s_data = subdata(data, d)
        compute_log_likelihood_variants(x_pars,  s_data, lambda, epsilon_J, epsilon_sigma, output_root_d)
        compute_mean_covariance(x_pars, s_data, lambda, epsilon_J, epsilon_sigma, output_root_d)
        
    end

   
end



        