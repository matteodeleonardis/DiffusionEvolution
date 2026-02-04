function run_analysis_pse1_null_model(;d, opt_pkg, output_root, contacts_file)
    # d=2
    # opt_pkg=:Optim
    # output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test"
    # contacts_file="/home/matteo/Projects/mDHFR/contacts_prediction/contact_map.npy"

    data_dir = joinpath(@__DIR__, "../../data")

    # experiment data
    file0 = joinpath(data_dir, "pse1/PSE1.fas")
    file1 = joinpath(data_dir, "pse1/Rnd10.fas")
    file2 = joinpath(data_dir, "pse1/Rnd20_init.fas")

    times = [10, 20]


    # natural sequences
    file_nat = joinpath(data_dir, "pse1/PSE1_clean.fasta")
    #file_nat = file1

    #training
    whiten = false
    extreme = false

    data, pca = process_data(file_nat, file0, [file1, file2], times; whiten=whiten, weight=false, d=d, extreme=extreme)

    ratio_tvar = pca.tprinvar / pca.tvar

    #save parameters and settings
    @save output_root * ".pars.jld2" ratio_tvar

    #parameters as tensors
    fasta_wt = readfasta(file0)
    wt = aa2int.(uppercase(fasta_wt[1][2]))
    L = length(fasta_wt[1][2])
    A = 21
    J_tens, h_tens, gamma = get_params_tens_null(data.x0, pca, d, A, L; whiten=whiten, eps_warning=1.0e-4, set_zero=false)

    #computing scores
    frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc = compute_scores(J_tens, h_tens, wt, L, output_root)

    true_contacts=JLD2.load(contacts_file)["contacts"]

    #ppv
    compute_ppv(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)

    #contact plot
    print_contact_plot(frobenius_score_zerosumgauge, frobenius_score_zerosumgauge_apc, 
        frobenius_score_wildtypegauge, frobenius_score_wildtypegauge_apc, true_contacts, L, output_root)
end

# d = parse(Int, ARGS[1])
# output_root = ARGS[2]

# run_training(d=d, opt_pkg=:Optim, output_root=output_root,
#     contacts_file=joinpath(data_dir, "pse1/contact_map.jld2")
