import Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using DiffusionEvolution

function collect_args()
    s = ArgParseSettings()
    @add_arg_table! s begin

        "--run"
            range_tester = x -> x in ["train", "plmdca", "low_rank_mf", "evcouplings", "random_seq_analysis", "reprint_inference",  "J_divergence"]
            help = "Which analysis to run. Options: train, plmdca, low_rank_mf, evcouplings, random_seq_analysis, reprint_inference, J_divergence"
            required = true

        "--data"
            range_tester = x -> x in ["dhfr", "pse1"]
            help = "Which data set to use. Options: dhfr, pse1"
            required = true

        "--dims"
            arg_type = Int
            nargs = '+'
            help = "Number of dimensions to use for the analysis."
            default = Int[]

        "--dir"
            help = "Directory with data to use"
            default = ""
            required = true

        "--gamma"
            range_tester = x -> x in ["yes", "no", ""]
            help = "Whether to learn gamma or not. Options: yes, no"
            default = ""

    end
    return parse_args(ARGS, s)
end

function execute()

    args = collect_args()
    DiffusionEvolution.main(args)
end

execute()
