# DiffusionEvolution: reproducing the paper figures

## 1. Install Julia 1.12.3

The analyses in the paper were performed using **Julia 1.12.3**. Use this version to reproduce the computational environment.

The instructions below are for Linux and macOS. Run the commands in a terminal.

### Install Juliaup

[Juliaup](https://github.com/JuliaLang/juliaup) is Julia's installer and version manager. If Juliaup is not already installed, run:

```bash
curl -fsSL https://install.julialang.org | sh
```

Follow the installer prompts, then close and reopen the terminal so that `juliaup` and `julia` are available on your `PATH`.

### Install and select Julia 1.12.3

```bash
juliaup add 1.12.3
juliaup default 1.12.3
```

The second command makes Julia 1.12.3 the default version launched by `julia`.

### Verify the installation

```bash
julia --version
```

Expected output:

```text
julia version 1.12.3
```

If you prefer to keep another Julia version as your default, omit `juliaup default 1.12.3` and explicitly select this version with:

```bash
julia +1.12.3
```

In that case, replace `julia` with `julia +1.12.3` in subsequent commands in this guide.

Installation reference: [official Julia installation documentation](https://docs.julialang.org/en/v1/manual/installation/).

## 2. Install the Julia packages

The Julia dependencies required by DiffusionEvolution are listed in `Project.toml`. From the root directory of the downloaded repository, install them with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The `--project=.` option activates the Julia environment defined by the `Project.toml` file in the current directory. `Pkg.instantiate()` downloads and installs the declared packages, and `Pkg.precompile()` precompiles them.

Verify that the package and its dependencies load correctly:

```bash
julia --project=. -e 'using DiffusionEvolution; println("DiffusionEvolution loaded successfully.")'
```

The expected final line is:

```text
DiffusionEvolution loaded successfully.
```

Run subsequent Julia commands from the repository root with `--project=.` so that they use this environment.

## 3. Install the Python environment

The Python dependencies and command-line tools used by the preprocessing code are listed in `diffusion_evolution.yml`. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if it is not already available, then run the following command from the repository root:

```bash
conda env create -f diffusion_evolution.yml
```

This creates a Conda environment named `diffusion_evolution` containing Python 3.10, NumPy, Biopython 1.78, HMMER, and tmux.

Activate the environment:

```bash
conda activate diffusion_evolution
```

Verify the Python packages and HMMER installation:

```bash
python --version
python -c 'import numpy, Bio; print("Python dependencies loaded successfully.")'
jackhmmer -h
```

The Python version should be 3.10, the second command should print `Python dependencies loaded successfully.`, and the last command should display the `jackhmmer` help text.

Activate this environment again with `conda activate diffusion_evolution` whenever you open a new terminal before running the preprocessing scripts.

## 4. Download the data and analysis results

The data and precomputed analysis results required to reproduce the paper figures can be downloaded from the following Google Drive folder:

[Download the DiffusionEvolution data and results](https://drive.google.com/drive/folders/1YjQJUz5-LQU3ZTx-tvbhEfKLvERdD4Cj?usp=sharing)

Download `data.zip` and `results_paper.zip`, then extract both archives.

`data.zip` contains a directory named `data`. Place this directory in the root of the repository, replacing the existing `DiffusionEvolution/data/` directory.

`results_paper.zip` contains a directory named `results`. Place this directory directly in the root of the repository as `DiffusionEvolution/results/`.

After extracting and moving the directories, the relevant part of the repository should have the following structure:

```text
DiffusionEvolution/
├── data/
├── results/
├── scripts/
├── src/
├── Project.toml
└── diffusion_evolution.yml
```

## 5. Reproduce the paper figures

The scripts used to generate the paper figures are organized by data set:

- `scripts/figures/dhfr_figures/` contains the mDHFR figure scripts.
- `scripts/figures/pse1_figures/` contains the PSE1 figure scripts.

The older `dhfr_data.jl` and `pse1_data.jl` scripts are not used. Use `dhfr_data_v2.jl` and `pse1_data_v2.jl` instead.

### Install the figure-script dependencies

The figure scripts have their own Julia environment in `scripts/figures/Project.toml`. From the repository root, connect this environment to the local DiffusionEvolution package and install its dependencies:

```bash
julia --project=scripts/figures -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'
```

### Run the scripts

Run each script from its own figure directory because the scripts activate the Julia environment in the parent `scripts/figures/` directory. For example:

```bash
cd scripts/figures/dhfr_figures
julia dhfr_data_v2.jl
```

and:

```bash
cd scripts/figures/pse1_figures
julia pse1_data_v2.jl
```

The generated SVG and PNG files are written to the current figure directory, except where a script explicitly constructs another output path.

Several scripts currently contain absolute input paths beginning with `/home/.../DiffusionEvolution`. Before running them on another computer, replace these paths with the location of the local repository. The scripts that compare fitted models read the precomputed files installed under `results/` in the previous section.

### mDHFR scripts

| Script | Figure produced |
|---|---|
| `dhfr_data_v2.jl` | Projects the mDHFR wild type and experimental rounds onto the first two principal components obtained from the natural sequences. It plots the weighted two-dimensional distribution at each round and marks the wild-type sequence. Output: `dhfr_latent_representation_v2.svg`. |
| `pca_dhfr.jl` | Computes two-dimensional PCA coordinates and kernel-density estimates for the natural and experimental mDHFR sequences, then overlays their densities. Output: `pca_dhfr_density.svg`. |
| `dhfr_cum_variance.jl` | Reads the fraction of total natural-sequence variance explained by each fitted latent dimension and plots it as a function of `d`. Output: `dhfr_cum_variance.svg`. |
| `dhfr_J_divergence.jl` | Calculates the relative off-diagonal weight, `rho`, of the inferred latent interaction matrix for each value of `d`. Output: `dhfr_off_diag_energy_tri.svg`. |
| `dhfr_cov_empirical_vs_model.jl` | Compares the empirical covariance at the last experimental round with the model covariance at that time and with the equilibrium covariance. It reports the Pearson correlation and mean squared difference across values of `d`, with element-by-element scatter plots. Outputs: `mdhfr_cov_empirical_vs_model.png` and `mdhfr_cov_empirical_vs_model.scatter.png`. |
| `dhfr_plmdca.jl` | Evaluates PlmDCA contact predictions against the mDHFR structural contact map. It produces the positive predictive value curve and a contact map showing correct and incorrect predictions. Outputs: `mdhfr_plmdca_ppv.svg` and `mdhfr_plmdca_contact.svg`. |
| `dhfr_low_rank.jl` | Compares contact predictions from the OU model, the PCA-based low-rank model, PlmDCA, and a random baseline across values of `d`. It also counts correct OU and low-rank predictions absent from the PlmDCA predictions, separated by sequence distance. Outputs: `mdhfr_n_contacts_comparison.svg` and `mdhfr_n_new_contacts_vs_plmdca.svg`. |
| `dhfr_evcouplings.jl` | Compares the OU contact predictions with the combined-MSA EVCouplings results and PlmDCA. It shows correct combined-MSA contacts absent from PlmDCA, compares the number of correct predictions, and compares additional OU and combined-MSA contacts relative to PlmDCA. Outputs: `mdhfr_evcouplings_contacts_vs_plmdca.svg`, `mdhfr_evcouplings_n_contacts_compare.svg`, and `mdhfr_evcouplings_new_contacts_ou_vs_combined_msa.svg`. |
| `dhfr_ancestor_pvalue.jl` | Reads trajectory-reconstruction p-values and displays a significance map over latent dimension `d` and ancestor time, using `p < 0.05` as the significance threshold. The result directory identifies whether the final experimental round was included in the training data, while `{type}` can be `uniform`, `site_mut`, or `profile`. Supply the selected file as the first command-line argument. |

For mDHFR, the trajectory-reconstruction results are organized as follows:

- `results/mdhfr/run0_pfam/` contains results from models trained using all experimental rounds.
- `results/mdhfr/run0_lastout/` contains results from models trained without the final experimental round.

Both directories contain files named `trajectory_reconstruction_pvalue_{type}.txt`, where `{type}` is `uniform`, `site_mut`, or `profile`. For example:

```bash
julia dhfr_ancestor_pvalue.jl ../../../results/mdhfr/run0_pfam/trajectory_reconstruction_pvalue_uniform.txt
julia dhfr_ancestor_pvalue.jl ../../../results/mdhfr/run0_lastout/trajectory_reconstruction_pvalue_uniform.txt
```

Replace `uniform` with `site_mut` or `profile` to generate the corresponding significance map.

### PSE1 scripts

| Script | Figure produced |
|---|---|
| `pse1_data_v2.jl` | Projects the PSE1 wild type and experimental rounds onto the first two principal components obtained from the natural sequences. It plots the weighted two-dimensional distribution at each round and marks the wild-type sequence. Output: `pse1_latent_representation_v2.svg`. |
| `pca_pse1.jl` | Computes two-dimensional PCA coordinates and kernel-density estimates for the natural and experimental PSE1 sequences, then overlays their densities. Output: `pca_pse1.svg`. |
| `pse1_cum_variance.jl` | Reads the fraction of total natural-sequence variance explained by each fitted latent dimension and plots it as a function of `d`. Output: `pse1_cum_variance.svg`. |
| `pse1_J_divergence.jl` | Calculates the relative off-diagonal weight, `rho`, of the inferred latent interaction matrix for each value of `d`. Output: `pse1_off_diag_energy_tri.svg`. |
| `pse1_cov_empirical_vs_model.jl` | Compares the empirical covariance at the last experimental round with the model covariance at that time and with the equilibrium covariance. It reports the Pearson correlation and mean squared difference across values of `d`, with element-by-element scatter plots. Outputs: `pse1_cov_empirical_vs_model.png` and `pse1_cov_empirical_vs_model.scatter.png`. |
| `pse1_plmdca.jl` | Evaluates PlmDCA contact predictions against the PSE1 structural contact map. It produces the positive predictive value curve and a contact map showing correct and incorrect predictions. Outputs: `pse1_plmdca_ppv.svg` and `pse1_plmdca_contact.svg`. |
| `pse1_low_rank.jl` | Compares contact predictions from the OU model, the PCA-based low-rank model, PlmDCA, and a random baseline across values of `d`. It also counts correct OU and low-rank predictions absent from the PlmDCA predictions, separated by sequence distance. Outputs: `pse1_n_contacts_comparison.svg` and `pse1_n_new_contacts_vs_plmdca.svg`. |
| `pse1_evcouplings.jl` | Compares the OU contact predictions with the combined-MSA EVCouplings results and PlmDCA. It shows correct combined-MSA contacts absent from PlmDCA, compares the number of correct predictions, and compares additional OU and combined-MSA contacts relative to PlmDCA. Outputs: `pse1_evcouplings_contacts_vs_plmdca.svg`, `pse1_evcouplings_n_contacts_compare.svg`, and `pse1_evcouplings_new_contacts_ou_vs_combined_msa.svg`. |
| `pse1_ancestor_pvalue.jl` | Reads trajectory-reconstruction p-values and displays a significance map over latent dimension `d` and ancestor time, using `p < 0.05` as the significance threshold. The result directory identifies whether the final experimental round was included in the training data, while `{type}` can be `uniform`, `site_mut`, or `profile`. Supply the selected file as the first command-line argument. |

For PSE1, the trajectory-reconstruction results are organized as follows:

- `results/pse1/run0/` contains results from models trained using all experimental rounds.
- `results/pse1/run0_lastout/` contains results from models trained without the final experimental round.

Both directories contain files named `trajectory_reconstruction_pvalue_{type}.txt`, where `{type}` is `uniform`, `site_mut`, or `profile`. For example:

```bash
julia pse1_ancestor_pvalue.jl ../../../results/pse1/run0/trajectory_reconstruction_pvalue_uniform.txt
julia pse1_ancestor_pvalue.jl ../../../results/pse1/run0_lastout/trajectory_reconstruction_pvalue_uniform.txt
```

Replace `uniform` with `site_mut` or `profile` to generate the corresponding significance map.
