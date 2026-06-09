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
                session="$(args["dir"])-mdhfr_d_$(d)"
                outdir="$(@__DIR__)/../../results/mdhfr/$(args["dir"])/mdhfr_analysis_d_$(d)"
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
                    julia --project=$(env_path) dhfr_analysis.jl $(d) $(output_root) \\
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
                "results", "mdhfr",
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



    elseif args.run == "low_rank_mf"

    elseif args.run == "evcouplings"

    elseif args.run == "random_seq_analysis"

    elseif args.run == "reprint_inference"

    elseif args.run == "J_divergence"

    end

end