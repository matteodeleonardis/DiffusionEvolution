function natural_sort_key(path::AbstractString)
    parts = split(path, r"(\d+)", keepempty=false)

    return map(parts) do part
        if occursin(r"^\d+$", part)
            return (0, parse(Int, part))
        else
            return (1, part)
        end
    end
end

function extract_d_value(path::AbstractString)
    m = match(r"_analysis_d_([0-9]+)", path)

    if m === nothing
        return typemax(Int)
    end

    return parse(Int, m.captures[1])
end


function main(args)

    if args["run"] == "train"
        if length(args["dims"]) == 0
            println("Specifify at least one dimension.")
            return
        end

        max_threads = "12"
        env_path="$(@__DIR__)/.."

        if args["data"] == "dhfr"
            for d in args["dims"]
                session="$(args["dir"])-mdhfr_d_$(d)"
                outdir="$(@__DIR__)/../results/mdhfr/$(args["dir"])/mdhfr_analysis_d_$(d)"
                output_root="$(outdir)/mdhfr_analysis_d_$(d)"

                stdout_log="$(outdir)/stdout.log"
                stderr_log="$(outdir)/stderr.log"

                mkpath(outdir)

                inner_cmd = """
                    OMP_NUM_THREADS=$(max_threads) \\
                    OPENBLAS_NUM_THREADS=$(max_threads) \\
                    MKL_NUM_THREADS=$(max_threads) \\
                    VECLIB_MAXIMUM_THREADS=$(max_threads) \\
                    JULIA_NUM_THREADS=1 \\
                    julia --project=$(env_path) $(@__DIR__)/../scripts/mdhfr/dhfr_analysis.jl $(d) $(output_root) \\
                    > $(stdout_log) 2> $(stderr_log)
                """

                run(`tmux new-session -d -s $session $inner_cmd`)
            end

        elseif args["data"] == "pse1"
            for d in args["dims"]
                session="$(args["dir"])-pse1_d_$(d)"
                outdir="$(@__DIR__)/../results/pse1/$(args["dir"])/pse1_analysis_d_$(d)"
                output_root="$(outdir)/pse1_analysis_d_$(d)"

                stdout_log="$(outdir)/stdout.log"
                stderr_log="$(outdir)/stderr.log"

                mkpath(outdir)

                inner_cmd = """
                    OMP_NUM_THREADS=$(max_threads) \\
                    OPENBLAS_NUM_THREADS=$(max_threads) \\
                    MKL_NUM_THREADS=$(max_threads) \\
                    VECLIB_MAXIMUM_THREADS=$(max_threads) \\
                    JULIA_NUM_THREADS=1 \\
                    julia --project=$(env_path) $(@__DIR__)/../scripts/pse1/pse1_analysis.jl $(d) $(output_root) \\
                    > $(stdout_log) 2> $(stderr_log)
                """

                run(`tmux new-session -d -s $session $inner_cmd`)
            end
        end

    elseif args["run"] == "plmdca"
        if args["gamma"] == ""
            println("Specify wheather gamma is learned. Options: yes/no")
            return
        end

        dir_files=args["dir"]
        gamma=args["gamma"]
        if args["data"] == "dhfr"
            this_dir=@__DIR__
            input_fasta="$(this_dir)/../data/dhfr/mDHFR_clean.fasta"
            plmdca_dir="$(this_dir)/../results/mdhfr/plmdca"
            output_root="$(this_dir)/../results/mdhfr/$(dir_files)/plmdca"
            output_dir=dirname(output_root)
            mkpath(output_dir)

            score_link = "$(output_root).score.jld2"
            score_target = joinpath(plmdca_dir, "plmdca.score.jld2")

            if !islink(score_link) || !ispath(score_link)
                ispath(score_link) && rm(score_link; force=true)
                symlink(score_target, score_link)
            end

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=natural_sort_key)

            plmdca_analysis_script = joinpath(@__DIR__, "../scripts/mdhfr/plmdca_analysis.jl")

            cmd = `julia $plmdca_analysis_script $input_fasta $plmdca_dir $output_root $gamma $model_score_files`

            run(cmd)
        elseif args["data"] == "pse1"
            this_dir=@__DIR__
            input_fasta="$(this_dir)/../data/pse1/PSE1_clean.fasta"
            plmdca_dir="$(this_dir)/../results/pse1/plmdca"
            output_root="$(this_dir)/../results/pse1/$(dir_files)/plmdca"
            output_dir=dirname(output_root)
            mkpath(output_dir)

            score_link = "$(output_root).score.jld2"
            score_target = joinpath(plmdca_dir, "plmdca.score.jld2")

            if !islink(score_link) || !ispath(score_link)
                ispath(score_link) && rm(score_link; force=true)
                symlink(score_target, score_link)
            end

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=natural_sort_key)

            plmdca_analysis_script = joinpath(@__DIR__, "../scripts/pse1/plmdca_analysis.jl")

            cmd = `julia $plmdca_analysis_script $input_fasta $plmdca_dir $output_root $gamma $model_score_files`

            run(cmd)
        end



    elseif args["run"] == "low_rank_mf"
        dir_files = args["dir"]

        if args["data"] == "dhfr"
            this_dir = @__DIR__

            input_fasta = "$(this_dir)/../data/dhfr/mDHFR_clean.fasta"
            input_wt = "$(this_dir)/../data/dhfr/mDHFR.fasta"

            output_root = "$(this_dir)/../results/mdhfr/$(dir_files)/low_rank_mf"
            low_rank_dir = "$(this_dir)/../results/mdhfr/low_rank_mf"

            mkpath(low_rank_dir)

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            low_rank_analysis_script = joinpath(@__DIR__, "../scripts/mdhfr/low_rank_dhfr_analysis.jl")

            cmd = `julia $low_rank_analysis_script $input_fasta $input_wt $output_root $low_rank_dir $model_score_files`

            run(cmd)

        elseif args["data"] == "pse1"
            this_dir = @__DIR__

            input_fasta = "$(this_dir)/../data/pse1/PSE1_clean.fasta"
            input_wt = "$(this_dir)/../data/pse1/PSE1.fasta"

            output_root = "$(this_dir)/../results/pse1/$(dir_files)/low_rank_mf"
            low_rank_dir = "$(this_dir)/../results/pse1/low_rank_mf"

            mkpath(low_rank_dir)

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            low_rank_analysis_script = joinpath(@__DIR__, "../scripts/pse1/low_rank_pse1_analysis.jl")

            cmd = `julia $low_rank_analysis_script $input_fasta $input_wt $output_root $low_rank_dir $model_score_files`

            run(cmd)
        end

    elseif args["run"] == "evcouplings"
        dir_files = args["dir"]

        if args["data"] == "dhfr"
            this_dir = @__DIR__

            in_nat = "$(this_dir)/../data/dhfr/mDHFR_clean.fasta"
            in_wt = "$(this_dir)/../data/dhfr/mDHFR.fasta"
            in_r1 = "$(this_dir)/../data/dhfr/Round1_Q15_C10_aa.aln"
            in_r2 = "$(this_dir)/../data/dhfr/Round2_Q15_C10_aa.aln"
            in_r3 = "$(this_dir)/../data/dhfr/Round3_Q15_C10_aa.aln"
            in_r4 = "$(this_dir)/../data/dhfr/Round4_Q15_C10_aa.aln"
            in_r5 = "$(this_dir)/../data/dhfr/Round5_Q15_C10_aa.aln"
            in_r15 = "$(this_dir)/../data/dhfr/Gen15_aa.aln"

            output_root = "$(this_dir)/../results/mdhfr/$(dir_files)/ev_couplings"
            output_dir = dirname(output_root)
            mkpath(output_dir)

            score_link = "$(output_root).score.jld2"
            score_target = "$(this_dir)/../results/mdhfr/ev_couplings/ev_couplings.score.jld2"

            if !islink(score_link) || !ispath(score_link)
                ispath(score_link) && rm(score_link; force=true)
                symlink(score_target, score_link)
            end

            evc_score_dir = "$(this_dir)/../results/mdhfr/ev_couplings"
            mkpath(evc_score_dir)

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            ev_couplings_analysis_script = joinpath(@__DIR__, "../scripts/mdhfr/ev_couplings_analysis.jl")

            cmd = `julia $ev_couplings_analysis_script $in_nat $in_r1 $in_wt $in_r2 $in_r3 $in_r4 $in_r5 $in_r15 $evc_score_dir $output_root $model_score_files`

            run(cmd)

        elseif args["data"] == "pse1"
            this_dir = @__DIR__

            in_nat = "$(this_dir)/../data/pse1/PSE1_clean.fasta"
            in_wt = "$(this_dir)/../data/pse1/PSE1.fasta"

            # Replace these names if your PSE1 round/alignment files differ.
            in_r1 = "$(this_dir)/../data/pse1/Round1_Q15_C10_aa.aln"
            in_r2 = "$(this_dir)/../data/pse1/Round2_Q15_C10_aa.aln"
            in_r3 = "$(this_dir)/../data/pse1/Round3_Q15_C10_aa.aln"
            in_r4 = "$(this_dir)/../data/pse1/Round4_Q15_C10_aa.aln"
            in_r5 = "$(this_dir)/../data/pse1/Round5_Q15_C10_aa.aln"
            in_r15 = "$(this_dir)/../data/pse1/Gen15_aa.aln"

            output_root = "$(this_dir)/../results/pse1/$(dir_files)/ev_couplings"
            output_dir = dirname(output_root)
            mkpath(output_dir)

            score_link = "$(output_root).score.jld2"
            score_target = "$(this_dir)/../results/pse1/ev_couplings/ev_couplings.score.jld2"

            if !islink(score_link) || !ispath(score_link)
                ispath(score_link) && rm(score_link; force=true)
                symlink(score_target, score_link)
            end

            evc_score_dir = "$(this_dir)/../results/pse1/ev_couplings"
            mkpath(evc_score_dir)

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            ev_couplings_analysis_script = joinpath(@__DIR__, "../scripts/pse1/ev_couplings_analysis.jl")

            cmd = `julia $ev_couplings_analysis_script $in_nat $in_r1 $in_wt $in_r2 $in_r3 $in_r4 $in_r5 $in_r15 $evc_score_dir $output_root $model_score_files`

            run(cmd)
        end

    elseif args["run"] == "random_seq_analysis"
        dir_files = args["dir"]
        random_seq = args["random_seq"]
        delta_t = args["delta_t"]
        
        if args["data"] == "dhfr"
            this_dir = @__DIR__

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_pars_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".pars.jld2")
                        push!(model_pars_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_pars_files; by=extract_d_value)

            rand_seq_analysis_script = joinpath(@__DIR__, "../scripts/mdhfr/rand_seq_analysis_dhfr.jl")

            cmd = `julia $rand_seq_analysis_script $random_seq $delta_t $model_pars_files`

            run(cmd)

        elseif args["data"] == "pse1"
            this_dir = @__DIR__

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_pars_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".pars.jld2")
                        push!(model_pars_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_pars_files; by=extract_d_value)

            rand_seq_analysis_script = joinpath(@__DIR__, "../scripts/pse1/rand_seq_analysis_pse1.jl")

            cmd = `julia $rand_seq_analysis_script $random_seq $delta_t$model_pars_files`

            run(cmd)
        end

    elseif args["run"] == "reprint_inference"
        dir_files = args["dir"]

        if args["data"] == "dhfr"
            this_dir = @__DIR__

            output_root = "$(this_dir)/../results/mdhfr/$(dir_files)/mdhfr_analysis_d_"

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            reprint_inference_script = joinpath(@__DIR__, "../scripts/mdhfr/reprint_inference.jl")

            cmd = `julia $reprint_inference_script $output_root $model_score_files`

            run(cmd)

        elseif args["data"] == "pse1"
            this_dir = @__DIR__

            output_root = "$(this_dir)/../results/pse1/$(dir_files)/pse1_analysis_d_"

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            reprint_inference_script = joinpath(@__DIR__, "../scripts/pse1/reprint_inference.jl")

            cmd = `julia $reprint_inference_script $output_root $model_score_files`

            run(cmd)
        end

    elseif args["run"] == "J_divergence"
        dir_files = args["dir"]

        if args["data"] == "dhfr"
            this_dir = @__DIR__

            input_fasta = "$(this_dir)/../data/dhfr/mDHFR_clean.fasta"
            input_wt = "$(this_dir)/../data/dhfr/mDHFR.fasta"
            output_root = "$(this_dir)/../results/mdhfr/$(dir_files)/J_divergence"

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "mdhfr",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "mdhfr_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            j_divergence_script = joinpath(@__DIR__, "../scripts/mdhfr/J_divergence_dhfr.jl")

            cmd = `julia $j_divergence_script $input_fasta $input_wt $output_root $model_score_files`

            run(cmd)

        elseif args["data"] == "pse1"
            this_dir = @__DIR__

            input_fasta = "$(this_dir)/../data/pse1/PSE1_clean.fasta"
            input_wt = "$(this_dir)/../data/pse1/PSE1.fasta"
            output_root = "$(this_dir)/../results/pse1/$(dir_files)/J_divergence"

            search_root = normpath(joinpath(
                this_dir,
                "..",
                "results", "pse1",
                dir_files
            ))

            model_score_files = String[]

            for (root, dirs, files) in walkdir(search_root)
                for file in files
                    if startswith(file, "pse1_analysis_d_") &&
                    endswith(file, ".scores.zerosumgauge_apc.tsv")
                        push!(model_score_files, joinpath(root, file))
                    end
                end
            end

            sort!(model_score_files; by=extract_d_value)

            j_divergence_script = joinpath(@__DIR__, "../scripts/pse1/J_divergence_pse1.jl")

            cmd = `julia $j_divergence_script $input_fasta $input_wt $output_root $model_score_files`

            run(cmd)
        end

    end

end