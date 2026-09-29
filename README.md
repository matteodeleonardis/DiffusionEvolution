# DiffusionEvolution analysis pipeline

Follow this steps to produce the results of the analysis from scratch. Anyway, to reproduce exactly the same figures of the paper, these are already computed and can be downloaded separately. See the `DiffusionEvolution_reproduction_guide.md`.

## Setup and command conventions

Run the commands below from the repository root:

```bash
cd DiffusionEvolution
```

The repository provides a Conda environment for Python, NumPy, Biopython, HMMER
and tmux:

```bash
conda env create -f diffusion_evolution.yml
conda activate diffusion_evolution
```

Julia must also be available on `PATH`. Install the Julia project dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The entry point activates the repository's Julia environment automatically.
Invoke it with Julia:

```bash
julia bin/cli.jl --help
```

### Command-line arguments

| Argument | Accepted values and actual use |
|---|---|
| `--run` | Required. Selects `train`, `J_divergence`, `plmdca`, `low_rank_mf`, `evcouplings`, `random_seq_analysis`, or `reprint_inference`. |
| `--data` | Required. Dataset identifier: `dhfr` or `pse1`. |
| `--dir` | Required for every mode. Run-directory name, such as `run_example`. This directory is assumed to be placed in `DiffusionEvolution/results/{data}/run_example`. Use the same name to analyse an existing training run. Don't use spaces or periods. |
| `--dims` | One or more space-separated positive integers, for example `--dims 10 20 30`. This is required only for training. |
| `--gamma` | Accepts `yes` or `no`; omitted value defaults to an empty string. The analysis in the paper has been made using `--gamma no`, so use this when required. `--gamma yes ` is still work-in-progress. |
| `--random_seq` | `uniform`, `profile`, or `site_mut`. Supply this explicitly for `random_seq_analysis`. |

## 1. Preprocess the natural sequence alignments

The scripts are located in `data/msa_preprocess/`. They align natural sequences
to the reference sequence with `jackhmmer`, convert the alignment to FASTA,
retain reference-matching columns, and filter incomplete or invalid sequences.

| Dataset | Reference sequence | Natural-sequence input | Clean alignment |
|---|---|---|---|
| DHFR | `data/dhfr/mDHFR.fasta` | `data/dhfr/PF00186.fasta` | `data/dhfr/mDHFR_clean.fasta` |
| PSE-1 | `data/pse1/PSE1.fas` | `data/pse1/PF13354.fasta` | `data/pse1/PSE1_clean.fasta` |

Run the following commands to preprocess data

```bash
# DHFR
cd data/msa_preprocess 
bash preprocess_dhfr.sh

# PSE-1
cd data/msa_preprocess
bash preprocess_pse1.sh
```

This step prepares the natural alignments used for PCA and the comparison
analyses. Experimental-round alignments and reference contact maps must already
be available separately.

## 2. Train the OU models

Train one model for each requested latent dimension:

```bash
# DHFR
julia bin/cli.jl --run train --data dhfr --dir run_example --dims 10 20 30 --gamma no

# PSE-1
julia bin/cli.jl --run train --data pse1 --dir run_example --dims 10 20 30 --gamma no
```

The training dispatcher launches one detached `tmux` session per dimension.
The command returns after launching the sessions, while training continues in
the background. Each session currently allows up to 12 BLAS/OpenMP threads and
sets one Julia thread, so several dimensions can consume substantial resources
concurrently.

Check the sessions with:

```bash
tmux ls
```

For DHFR, each dimension has its own directory:

```text
results/mdhfr/run_example/mdhfr_analysis_d_<d>/
```

For PSE-1:

```text
results/pse1/run_example/pse1_analysis_d_<d>/
```

These directories contain `stdout.log`, `stderr.log`, and model outputs with
the corresponding `mdhfr_analysis_d_<d>` or `pse1_analysis_d_<d>` prefix.
Outputs include:

- `.pars.jld2`: fitted parameters, the empirical gamma estimate, and the
  fraction of natural-sequence variance retained by PCA.
- `.settings.jld2`: model settings.
- `.optimization.log`: optimization results and diagnostics.
- `.scores.zerosumgauge_apc.tsv`: contact scores used by downstream analyses.
- Contact-prediction evaluation files and plots.

Wait for all requested models to finish and inspect their logs before running
the downstream analyses. Check the `*optimization.log` files to see if the optimization ended successfully.

### Training configuration in the checked version

The training settings are defined in
`src/analysis/train_and_analysis_dhfr.jl` and
`src/analysis/train_and_analysis_pse1.jl`.

| Dataset | Experimental rounds included in training | Final round excluded |
|---|---|---|
| DHFR | 1, 2, 3, 4, 5 | 15 |
| PSE-1 | 10 | 20 |

At this moment, the CLI has no option for selecting training rounds. Because the code is intended to be used automatically in the most standard way. If you want to repeat the held-out analysis from the paper, modify the array `times` inside `src/analysis/train_and_analysis_dhfr.jl` or `src/analysis/train_and_analysis_dhfr.jl`.

## 3. Model selection: rho analysis

Run the analysis across the trained dimensions in the selected run:

```bash
julia bin/cli.jl --run J_divergence --data dhfr --dir run_example
```

For PSE-1, replace `--data dhfr` with `--data pse1`. This substitution also
applies to all subsequent examples.


## 4. plmDCA analysis

Compare OU contact predictions with plmDCA applied to the natural alignment:

```bash
julia bin/cli.jl --run plmdca --data dhfr --dir run_example --gamma no
```

The shared score cache is stored at:

- DHFR: `results/mdhfr/plmdca/plmdca.score.jld2`.
- PSE-1: `results/pse1/plmdca/plmdca.score.jld2`.


## 5. Low-rank Gaussian approximation analysis

After completing plmDCA for the same run, execute:

```bash
julia bin/cli.jl --run low_rank_mf --data dhfr --dir run_example
```

The routine requires `plmdca.score.jld2` in the run directory and returns
without completing the analysis if it is absent.

Low-rank scores are cached under `results/mdhfr/low_rank_mf/` or
`results/pse1/low_rank_mf/`. 

## 6. Combined-MSA analysis

After completing plmDCA for the same run, execute:

```bash
julia bin/cli.jl --run evcouplings --data dhfr --dir run_example
```

Shared scores are stored under `results/mdhfr/ev_couplings/` or
`results/pse1/ev_couplings/`. Run-specific outputs use the prefix
`ev_couplings`.

**Current PSE-1 cache issue:** when cached combined-MSA scores already exist,
`src/analysis/evcouplings_pse1.jl` loads them using
`evc_score_dir * "ev_couplings.score.jld2"`. The dispatcher supplies a directory
without a trailing slash, so this constructs the wrong path. The cache-loading
path must use `joinpath(evc_score_dir, "ev_couplings.score.jld2")` for reruns to
load the saved file correctly.

## 7. Random-sequence analysis and trajectory reconstruction

Select the trained-model run and the random comparison model:

```bash
julia bin/cli.jl --run random_seq_analysis --data dhfr --dir run_example --random_seq uniform
julia bin/cli.jl --run random_seq_analysis --data dhfr --dir run_example --random_seq profile
julia bin/cli.jl --run random_seq_analysis --data dhfr --dir run_example --random_seq site_mut
```

Each command runs one comparison type. The accepted values represent these
statistical null models:

| Value | Comparison model |
|---|---|
| `uniform` | A common mutation probability across sites, estimated separately for each experimental round; alternative amino acids are sampled uniformly. |
| `profile` | Independent sampling at each site from the observed, count-weighted amino-acid frequencies for each round. |
| `site_mut` | Site-specific mutation probabilities estimated separately for each round; alternative amino acids are sampled uniformly. |

The analysis compares experimental and random sequences through Hamming-distance
distributions, projected means and covariances, variant likelihoods, and
ancestor/trajectory-reconstruction likelihoods. It writes per-model figures
beside the parameter files and run-level summaries, including:

- `backward_ancestor_reconstruction_pvalue_<type>.txt`
- `forward_ancestor_reconstruction_pvalue_<type>.txt`
- `trajectory_reconstruction_pvalue_<type>.txt`

The summary files contain dimension, time, and p-value columns, and have
corresponding PNG plots. The random-sequence type appears in the output names.
