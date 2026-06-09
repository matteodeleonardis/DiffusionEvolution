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

    elseif args.run == "plmdca"

    elseif args.run == "low_rank_mf"

    elseif args.run == "evcouplings"

    elseif args.run == "random_seq_analysis"

    elseif args.run == "reprint_inference"

    elseif args.run == "J_divergence"

    end

end