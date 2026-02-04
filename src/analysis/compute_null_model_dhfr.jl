function run_analysis_dhfr_null_model(;d, opt_pkg, output_root, contacts_file)
    # d=2
    # opt_pkg=:Optim
    # output_root="/home/matteo/.julia/dev/DiffusionEvolution/analysis/test_results/test"
    # contacts_file="/home/matteo/Projects/mDHFR/contacts_prediction/contact_map.npy"

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
    #file_nat = file1

    #training
    whiten = false
    extreme = false

    data, pca = process_data(file_nat, file0, [file1, file2, file3, file4, file5, file15], times; whiten=whiten, weight=false, d=d, extreme=extreme)

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

    true_contacts=npzread(contacts_file)

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
#     contacts_file="/home/students/s301803/dhfr_neutral_evolution/DHFR/contact_map.npy")
