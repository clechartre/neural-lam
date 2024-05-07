"""Verification function to plot predictions and target data."""

# Standard library
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

# Third-party
import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# First-party
from neural_lam import constants, utils
from neural_lam.weather_dataset import WeatherDataModule


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def offline_plotting():
    """Plot expected output against npy prediction
    generated by the model.
    """
    parser = ArgumentParser(
        description="Standalone offline plotting of zarr and npy files"
    )

    parser.add_argument(
        "--path_target_file",
        type=str,
        default="data/cosmo/samples/predict/data.zarr",
        help="Path to the .zarr archive to verify against - target",
    )
    parser.add_argument(
        "--path_prediction_file",
        type=str,
        default="templates/predictions.npy",
        help="Path to the file output from the inference as .npy",
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        default="/users/clechart/neural-lam/figures/",
        help="Path to save the graphical output of this function",
    )
    parser.add_argument(
        "--variable_to_plot",
        type=str,
        default="QV",
        help="Variable to plot in short format",
    )
    parser.add_argument(
        "--level_to_plot",
        type=int,
        default=1,
        help="For 3D variables, which vertical level to plot?",
    )

    # Get args
    args = parser.parse_args()

    # # Mapping out the feature channel to its index
    mapping_dictionary = precompute_variable_indices()
    # Here we can select the level
    feature_channel = mapping_dictionary[args.variable_to_plot][0]
    if constants.IS_3D[args.variable_to_plot]:
        # We need to select the right level in constants.VERTICAL_LEVELS
        try:
            index = constants.VERTICAL_LEVELS.index(args.level_to_plot)
        except ValueError as exc:
            raise ValueError(
                f"The level {args.level_to_plot} is not valid."
                f"Choose from {constants.VERTICAL_LEVELS}"
            ) from exc
        feature_channel = mapping_dictionary[args.variable_to_plot][index]

    # Load inference dataset
    predictions_data_module = WeatherDataModule(
        "cosmo",
        path_verif_file=args.path_prediction_file,
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
    time_steps = generate_time_steps()

    # Load target dataset, only select the relevant time range
    target_all = xr.open_zarr(
        args.path_target_file,
        consolidated=True,
    )[
        args.variable_to_plot
    ].isel(time=slice(0, len(time_steps)))

    # Get Lon and Lat coordinates
    lon = target_all.lon.values
    lat = target_all.lat.values
    vmin = target_all.min().values
    vmax = target_all.max().values

    # We need to only select for a level?
    for i, time_step in time_steps.items():

        # Select the time step
        target = target_all.isel(time=i)
        # Convert target data to NumPy array and select right level here
        if constants.IS_3D[args.variable_to_plot]:
            target = target[feature_channel]

        target_tensor = torch.tensor(target.values)
        target_array = target_tensor.reshape(
            constants.GRID_SHAPE[0], constants.GRID_SHAPE[1]
        )
        target_feature_array = np.array(target_array)

        # Convert predictions to NumPy array
        prediction_array = (
            predictions[0, i, :, feature_channel]
            .reshape(*constants.GRID_SHAPE[::-1])
            .moveaxis([-2, -1], [-1, -2])
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

        # Titles for each subplot
        titles = ["Ground Truth", "Prediction"]

        # Plot each dataset
        for index, (ax, data) in enumerate(
            zip(axes, (target_feature_array, prediction_array))
        ):
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
            # Set specific title for each subplot
            ax.set_title(
                f"{titles[index]}: {args.variable_to_plot} at time step {i}"
            )

        # Add colorbar and titles
        cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
        cbar.ax.tick_params(labelsize=10)

        # Save plot
        directory = os.path.dirname(args.saving_path)
        plot = (
            f"{args.saving_path}feature_channel_"
            f"{feature_channel}_{time_step}.png"
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(
            plot,
            bbox_inches="tight",
        )
        plt.close()


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


def generate_time_steps():
    """Generate a list with all time steps in inference."""
    # Parse the times
    base_time = constants.EVAL_DATETIMES[0]
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y%m%d%H")
    time_steps = {}
    # Generate dates for each step
    for i in range(constants.EVAL_HORIZON - 2):
        # Compute the new date by adding the step interval in hours - 3
        new_date = base_time + timedelta(hours=i * constants.TRAIN_HORIZON)
        # Format the date back
        time_steps[i] = new_date.strftime("%Y%m%d%H")

    return time_steps


# Entry point for script
if __name__ == "__main__":
    offline_plotting()  # pylint: disable=unused-argument
