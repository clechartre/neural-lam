# Neural-LAM: Neural Weather Prediction for Limited Area Modeling

## Preparation

This project has been created from the
[MeteoSwiss Python blueprint](https://github.com/MeteoSwiss-APN/mch-python-blueprint)
for the CSCS.
The recommended way to manage Python versions is with `Conda`
(https://docs.conda.io/en/latest/).
On CSCS machines it is recommended to install the leaner `Miniconda`
(https://docs.conda.io/en/latest/miniconda.html),
which offers enough functionality for most of our use cases.
If you don't want to do this step manually, you may use the script
`tools/setup_miniconda.sh`.
The default installation path of this script is the current working directory,
you might want to change that with the `-p` option to a common location for all
environments, like e.g. `$SCRATCH`. If you want the script to immediately
initialize conda (executing `conda init` and thereby adding a few commands at the
end of your `.bashrc`) after installation, add the `-u` option:

```bash
tmpl/tools/setup_miniconda.sh -p $SCRATCH -u
```

In case you ever need to uninstall miniconda, do the following:

```bash
conda init --reverse --all
rm -rf $SCRATCH/miniconda
```

## Start developing

Once you created or cloned this repository, make sure the installation is running properly. Install the package dependencies with the provided script `setup_env.sh`.
Check available options with
```bash
tools/setup_env.sh -h
```
We distinguish pinned installations based on exported (reproducible) environments and free installations where the installation
is based on top-level dependencies listed in `requirements/requirements.yml`. If you start developing, you might want to do an unpinned installation and export the environment:

```bash
tools/setup_env.sh -u -e -n <package_env_name>
```
*Hint*: If you are the package administrator, it is a good idea to understand what this script does, you can do everything manually with `conda` instructions.

*Hint*: Use the flag `-m` to speed up the installation using mamba. Of course you will have to install mamba first (we recommend to install mamba into your base
environment `conda install -c conda-forge mamba`. If you install mamba in another (maybe dedicated) environment, environments installed with mamba will be located
in `<miniconda_root_dir>/envs/mamba/envs`, which is not very practical.

The package itself is installed with `pip`. For development, install in editable mode:

```bash
conda activate <package_env_name>
pip install --editable .
```

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

Once your package is installed, run the tests by typing:

```
conda activate <package_env_name>
pytest
```

If the tests pass, you are good to go. If not, contact the package administrator Simon Adamov. Make sure to update your requirement files and export your environments after installation
every time you add new imports while developing. Check the next section to find some guidance on the development process if you are new to Python and/or APN.

### Roadmap to your first contribution

Generally, the source code of your library is located in `src/<library_name>`. The blueprint will generate some example code in `mutable_number.py`, `utils.py` and `cli.py`. `cli.py` thereby serves as an entry
point for functionalities you want to execute from the command line, it is based on the Click library. If you do not need interactions with the command line, you should remove `cli.py`. Moreover, of course there exist other options for command line interfaces,
a good overview may be found here (https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/), we recommend however to use click. The provided example
code should provide some guidance on how the individual source code files interact within the library. In addition to the example code in `src/<library_name>`, there are examples for
unit tests in `tests/<library_name>/`, which can be triggered with `pytest` from the command line. Once you implemented a feature (and of course you also
implemented a meaningful test ;-)), you are likely willing to commit it. First, go to the root directory of your package and run pytest.

```bash
conda activate <package_env_name>
cd <package-root-dir>
pytest
```

If you use the tools provided by the blueprint as is, pre-commit will not be triggered locally but only if you push to the main branch
(or push to a PR to the main branch). If you consider it useful, you can set up pre-commit to run locally before every commit by initializing it once. In the root directory of
your package, type:

```bash
pre-commit install
```

If you run `pre-commit` without installing it before (line above), it will fail and the only way to recover it, is to do a forced reinstallation (`conda install --force-reinstall pre-commit`).
You can also just run pre-commit selectively, whenever you want by typing (`pre-commit run --all-files`). Note that mypy and pylint take a bit of time, so it is really
up to you, if you want to use pre-commit locally or not. In any case, after running pytest, you can commit and the linters will run at the latest on the GitHub actions server,
when you push your changes to the main branch. Note that pytest is currently not invoked by pre-commit, so it will not run automatically. Automated testing can be set up with
GitHub Actions or be implemented in a Jenkins pipeline (template for a plan available in `jenkins/`. See the next section for more details.

## Development tools

As this package was created with the APN Python blueprint, it comes with a stack of development tools, which are described in more detail on
(https://meteoswiss-apn.github.io/mch-python-blueprint/). Here, we give a brief overview on what is implemented.

### Testing and coding standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in APN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS
machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

### Pre-commit on GitHub actions

`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned) on the GitHub actions server and
then run various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that)
and in pull requests to the main branch.

### Jenkins

A jenkinsfile is available in the `jenkins/` folder. It can be used for a multibranch jenkins project, which builds
both commits on branches and PRs. Your jenkins pipeline will not be set up
automatically. If you need to run your tests on CSCS machines, contact DevOps to help you with the setup of the pipelines. Otherwise, you can ignore the jenkinsfiles
and exclusively run your tests and checks on GitHub actions.

## Features

<p align="middle">
    <img src="figures/neural_lam_header.png" width="700">
</p>
Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
The code uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/pytorch-lightning).
Graph Neural Networks are implemented using [PyG](https://pyg.org/) and logging is set up through [Weights & Biases](https://wandb.ai/).

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

For more information see our preprint: [*Graph-based Neural Weather Prediction for Limited Area Modeling*](https://arxiv.org/abs/2309.17370).
If you use Neural-LAM in your work, please cite:
```
@article{oskarsson2023graphbased,
      title={Graph-based Neural Weather Prediction for Limited Area Modeling},
      author={Joel Oskarsson and Tomas Landelius and Fredrik Lindsten},
      year={2023},
      journal={arXiv preprint arXiv:2309.17370}
}
```

We plan to continue updating this repository as we improve existing models and develop new ones.
Collaborations around this implementation are very welcome.
If you are working with Neural-LAM feel free to get in touch and/or submit pull requests to the repository.

<span style="color:#98FF98;">Additions relevant to the COSMO Neural-LAM implementation are highlighted in __green__.</span>

# Modularity
The Neural-LAM code is designed to modularize the different components involved in training and evaluating neural weather prediction models.
Models, graphs and data are stored separately and it should be possible to swap out individual components.
Still, some restrictions are inevitable:

* The graph used has to be compatible with what the model expects. E.g. a hierarchical model requires a hierarchical graph.
* The graph and data are specific to the limited area under consideration. This is of course true for the data, but also the graph should be created with the exact geometry of the area in mind.

<p align="middle">
  <img src="figures/neural_lam_setup.png" width="600"/>
</p>


## A note on the limited area setting
Currently we are using these models on a limited area covering the Nordic region, the so called MEPS area (see [paper](https://arxiv.org/abs/2309.17370)).
There are still some parts of the code that is quite specific for the MEPS area use case.
This is in particular true for the mesh graph creation (`create_mesh.py`) and some of the constants used (`neural_lam/constants.py`).
If there is interest to use Neural-LAM for other areas it is not a substantial undertaking to refactor the code to be fully area-agnostic.
We would be happy to support such enhancements.
See the issues https://github.com/joeloskarsson/neural-lam/issues/2, https://github.com/joeloskarsson/neural-lam/issues/3 and https://github.com/joeloskarsson/neural-lam/issues/4 for some initial ideas on how this could be done.

<span style="color:#98FF98;">

For the COSMO implementation some additional settings can be defined in `neural_lam/constants`. Most of the code should take user input either from `neural_lam/constants` or directly from command-line argument parsing. Would certainly be worth the effort to make the code fully area-agnostic.

</span>

# Using Neural-LAM
Below follows instructions on how to use Neural-LAM to train and evaluate models.

## Installation

<span style="color:#98FF98;">

For COSMO we use conda to avoid the Cartopy installation issues and because conda environments usually work well on the vCluster called Balfrin.cscs.ch.

1. Simply run `conda env create -f environment.yml` to create the environment.
2. Activate the environment with `conda activate neural-lam`.
3. Happy Coding \o/

Note that only the cuda version is pinned to 11.8, otherwise all the latest libraries are installed. This might break in the future and must be adjusted to the users conda version.

</span>

\
Follow the steps below to create the neccesary python environment.

1. Install GEOS for your system. For example with `sudo apt-get install libgeos-dev`. This is neccesary for the Cartopy requirement.
2. Use python 3.9.
3. Install version 2.0.1 of PyTorch. Follow instructions on the [PyTorch webpage](https://pytorch.org/get-started/previous-versions/) for how to set this up with GPU support on your system.
4. Install required packages specified in `requirements.txt`.
5. Install PyTorch Geometric version 2.2.0. This can be done by running
```
TORCH="2.0.1"
CUDA="cu117"

pip install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1\
    torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
You will have to adjust the `CUDA` variable to match the CUDA version on your system or to run on CPU. See the [installation webpage](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more information.

## Data
Datasets should be stored in a directory called `data`.
See the [repository format section](#format-of-data-directory) for details on the directory structure.

The full MEPS dataset can be shared with other researchers on request, contact us for this.
A tiny subset of the data (named `meps_example`) is available in `example_data.zip`, which can be downloaded from [here](https://liuonline-my.sharepoint.com/:f:/g/personal/joeos82_liu_se/EuiUuiGzFIFHruPWpfxfUmYBSjhqMUjNExlJi9W6ULMZ1w?e=97pnGX).
Download the file and unzip in the neural-lam directory.
All graphs used in the paper are also available for download at the same link (but can as easily be re-generated using `create_mesh.py`).
Note that this is far too little data to train any useful models, but all scripts can be ran with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be ran without any issues.

<span style="color:#98FF98;">

For COSMO the data is stored in the `data` folder with the same structure, but called `cosmo`. The data will be open-source someday but for now we cannot share the data outside of our vCluster. A tiny example dataset could probably be made available.

</span>

## Pre-processing
An overview of how the different scripts and files depend on each other is given in this figure:
<p align="middle">
  <img src="figures/component_dependencies.png"/>
</p>
In order to start training models at least three pre-processing scripts have to be ran:

* `create_mesh.py`
* `create_grid_features.py`
* `create_parameter_weights.py`

<span style="color:#98FF98;">

For COSMO also run `create_static_features.py` to create the static features for the graph nodes.

</span>

### Create graph
Run `create_mesh.py` with suitable options to generate the graph you want to use (see `python create_mesh.py --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python create_mesh.py --graph multiscale`
* **Hi-LAM**: `python create_mesh.py --graph hierarchical --hierarchical 1` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python create_mesh.py --graph 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

### Create remaining static features
To create the remaining static files run the scripts `create_grid_features.py` and `create_parameter_weights.py`.
The main option to set for these is just which dataset to use.

## Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally to a directory like `wandb/dryrun...`.
The W&B project name is set to `neural-lam`, but this can be changed in `neural_lam/constants.py`.
See the [W&B documentation](https://docs.wandb.ai/) for details.

If you would like to login and use W&B, run:
```
wandb login
```
If you would like to turn off W&B and just log things locally, run:
```
wandb off
```

## Train Models
Models can be trained using `train_model.py`.
Run `python train_model.py --help` for a full list of training options.
A few of the key ones are outlined below:

* `--dataset`: Which data to train on
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps`: Number of time steps to unroll for when making predictions and computing the loss

<span style="color:#98FF98;">

For COSMO three simple slurm sbatch scripts are available for training/evaluating/debugging the model. You can launch either of these jobs respectively with:

```
sbatch slurm_train.sh
sbatch slurm_eval.sh
sbatch slurm_debug.sh
```

This will train the model using the same seed and data as seen in the figures on wandb.

</span>


Checkpoints of trained models are stored in the `saved_models` directory.
The implemented models are:

### Graph-LAM
This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step pedictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](https://arxiv.org/abs/2309.17370), only with different graphs.

To train 1L-LAM use
```
python train_model.py --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python train_model.py --model graph_lam --graph multiscale ...
```

### Hi-LAM
A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python train_model.py --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in paralell.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.

To train Hi-LAM-Parallel use
```
python train_model.py --model hi_lam_parallel --graph hierarchical ...
```

Checkpoint files for our models trained on the MEPS data are available upon request.

## Evaluate Models
Evaluation is also done using `train_model.py`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation (not `ar_steps`, evaluation always unrolls full forecasts).
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable. This issue stems from PyTorch Lightning. See for example [this draft PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion and ongoing work to remedy this.

# Repository Structure
Except for training and pre-processing scripts all the source code can be found in the `neural_lam` directory.
Model classes, including abstract base classes, are located in `neural_lam/models`.

## Format of data directory
It is possible to store multiple datasets in the `data` directory.
Each dataset contains a set of files with static features and a set of samples.
The samples are split into different sub-directories for training, validation and testing.
The directory structure is shown with examples below.
Script names within parenthesis denote the script used to generate the file.
```
data
├── dataset1
│   ├── samples                             - Directory with data samples
│   │   ├── train                           - Training data
│   │   │   ├── nwp_2022040100_mbr000.npy  - A time series sample
│   │   │   ├── nwp_2022040100_mbr001.npy
│   │   │   ├── ...
│   │   │   ├── nwp_2022043012_mbr001.npy
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy   - Solar flux forcing
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
│   │   │   ├── ...
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022043012.npy
│   │   │   ├── wtr_2022040100.npy          - Open water features for one sample
│   │   │   ├── wtr_2022040112.npy
│   │   │   ├── ...
│   │   │   └── wtr_202204012.npy
│   │   ├── val                             - Validation data
│   │   └── test                            - Test data
│   └── static                              - Directory with graph information and static features
│       ├── nwp_xy.npy                      - Coordinates of grid nodes (part of dataset)
│       ├── surface_geopotential.npy        - Geopotential at surface of grid nodes (part of dataset)
│       ├── border_mask.npy                 - Mask with True for grid nodes that are part of border (part of dataset)
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)
│       ├── flux_stats.pt                   - Mean and std.-dev. of solar flux forcing (create_parameter_weights.py)
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)
├── dataset2
├── ...
└── datasetN
```

## Format of graph directory
The `graphs` directory contains generated graph structures that can be used by different graph-based models.
The structure is shown with examples below:
```
graphs
├── graph1                                  - Directory with a graph definition
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── g2m_edge_index.pt                   - Edges from grid to mesh (create_mesh.py)
│   ├── m2g_edge_index.pt                   - Edges from mesh to grid (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── g2m_features.pt                     - Static features of grid to mesh edges (create_mesh.py)
│   ├── m2g_features.pt                     - Static features of mesh to grid edges (create_mesh.py)
│   └── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
├── graph2
├── ...
└── graphN
```

### Mesh hierarchy format
To keep track of levels in the mesh graph, a list format is used for the files with mesh graph information.
In particular, the files
```
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
```
all contain lists of length `L`, for a hierarchical mesh graph with `L` layers.
For non-hierarchical graphs `L == 1` and these are all just singly-entry lists.
Each entry in the list contains the corresponding edge set or features of that level.
Note that the first level (index 0 in these lists) corresponds to the lowest level in the hierarchy.

In addition, hierarchical mesh graphs (`L > 1`) feature a few additional files with static data:
```
├── graph1
│   ├── ...
│   ├── mesh_down_edge_index.pt             - Downward edges in mesh graph (create_mesh.py)
│   ├── mesh_up_edge_index.pt               - Upward edges in mesh graph (create_mesh.py)
│   ├── mesh_down_features.pt               - Static features of downward mesh edges (create_mesh.py)
│   ├── mesh_up_features.pt                 - Static features of upward mesh edges (create_mesh.py)
│   ├── ...
```
These files have the same list format as the ones above, but each list has length `L-1` (as these edges describe connections between levels).
Entries 0 in these lists describe edges between the lowest levels 1 and 2.

## Contact
If you are interested in machine learning models for LAM, have questions about our implementation or ideas for extending it, feel free to get in touch.
You can open a github issue on this page, or (if more suitable) send an email to [joel.oskarsson@liu.se](mailto:joel.oskarsson@liu.se).


This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://meteoswiss-apn.github.io/mch-python-blueprint/) project template.
