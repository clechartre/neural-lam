# Standard library
from argparse import ArgumentParser
import os

# Third-party
import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# First-party
from neural_lam import constants, utils
from neural_lam.rotate_grid import unrotate_latlon
from neural_lam.weather_dataset import WeatherDataModule


# Verification function to plot predictions and target data
@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def offline_plotting():
    parser = ArgumentParser(
        description="Standalone offline plotting of zarr and npy files"
    )

    parser.add_argument(
        "--path_target_file",
        type=str,
        default="/users/clechart/clechart/neural-lam/data/offline/data_2020011017.zarr",
        help="Path to the .zarr archive to verify against - target",
    )
    parser.add_argument(
        "--path_prediction_file",
        type=str,
        default="/users/clechart/clechart/neural-lam/wandb/run-20240411_140635-cux0r96n/files/results/inference/prediction_0.npy",
        help="Path to the file output from the inference as .npy",
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        default="/users/clechart/clechart/neural-lam/figures/",
        help="Path to save the graphical output of this function",
    )
    parser.add_argument(
        "--variable_to_plot",
        type=str,
        default="TQV",
        help="Variable to plot in short format",
    )
    
    # Get args
    args = parser.parse_args()
    
    # Mapping out the feature channel to its index 
    mapping_dictionary = precompute_variable_indices()
    feature_channel = mapping_dictionary[args.variable_to_plot][0]

    z_1 = constants.VERTICAL_LEVELS
    target_all = (
        xr.open_zarr(
            args.path_target_file,
            consolidated=True,
        )[args.variable_to_plot]
        # .isel(time=0)
        # .transpose("x_1", "y_1")
    )

    # Load inference dataset
    predictions_data_module = WeatherDataModule(
        "cosmo_old",
        path_verif_file=args.path_prediction_file,
        split="verif",
        standardize=False,
        subset=False,
        batch_size=6,
        num_workers=2,
    )
    predictions_data_module.setup(stage="verif")
    predictions_loader = predictions_data_module.verif_dataloader()
    for predictions_batch in predictions_loader:
        predictions = predictions_batch[0]
        break

    # Unrotate lat and lon coordinates
    lon, lat = unrotate_latlon(target_all)

    vmin = target_all.min().values
    vmax = target_all.max().values

    for i in range(22):

        target = target_all.isel(time=i).transpose("x_1", "y_1")
        # Convert target data to NumPy array
        target_tensor = torch.tensor(target.values)
        target_array = target_tensor.reshape(*constants.GRID_SHAPE[::-1])
        target_feature_array = np.array(target_array)

        # Convert predictions to NumPy array
        prediction_array = (
            predictions[0, i, :, feature_channel]
            .reshape(*constants.GRID_SHAPE[::-1])
            .cpu()
            .numpy()
        )

        # Create plot with two subplots for target and predictions
        fig, axes = plt.subplots(
            2,
            1,
            figsize=constants.FIG_SIZE,
            subplot_kw={"projection": constants.SELECTED_PROJ},
        )

        # Plot each dataset
        for ax, data in zip(axes, (target_feature_array, prediction_array)):
            contour_set = ax.contourf(
                lon,
                lat,
                data,
                transform=constants.SELECTED_PROJ,
                cmap="plasma",
                levels=np.linspace(vmin, vmax, num=100),
            )
            ax.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
            ax.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
            ax.gridlines(
                crs=constants.SELECTED_PROJ,
                draw_labels=False,
                linewidth=0.5,
                alpha=0.5,
            )
            ax.set_title(f"{args.variable_to_plot} at time step {i}")

            # Add colorbar and titles
            cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
            cbar.ax.tick_params(labelsize=10)
            plt.savefig(
                os.path.join(args.saving_path, f"plot_offline_{i}.png"), bbox_inches="tight"
            )
            plt.close(fig)


def precompute_variable_indices():
    """
    Precompute indices for each variable in the input tensor
    """
    variable_indices = {}
    all_vars = []
    index = 0
    # Create a list of tuples for all variables, using level 0 for 2D
    # variables
    for var_name in constants.PARAM_NAMES_SHORT:
        if constants.IS_3D[var_name]:
            for level in constants.VERTICAL_LEVELS:
                all_vars.append((var_name, level))
        else:
            all_vars.append((var_name, 0))  # Use level 0 for 2D variables

    # Sort the variables based on the tuples
    sorted_vars = sorted(all_vars)

    for var in sorted_vars:
        var_name, level = var
        if var_name not in variable_indices:
            variable_indices[var_name] = []
        variable_indices[var_name].append(index)
        index += 1

    return variable_indices


# Entry point for script
if __name__ == "__main__":
    offline_plotting() #pylint: disable = missing-arguments
