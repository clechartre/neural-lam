"""Verification function to plot predictions and target data."""

# Standard library
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# First-party
from neural_lam import constants, utils
from neural_lam.vis import (
    create_geographic_plot,
    load_verification_data,
    precompute_variable_indices,
)


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def offline_plotting():
    """
    Generate comparison plot between prediction array and ground truth.
    """
    parser = ArgumentParser(
        description="Standalone offline plotting of zarr and npy files"
    )
    # Command-line options
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

    args = parser.parse_args()

    mapping_dictionary = precompute_variable_indices()
    feature_channel = mapping_dictionary[args.variable_to_plot][0]
    if constants.IS_3D[args.variable_to_plot]:
        index = constants.VERTICAL_LEVELS.index(args.level_to_plot)
        feature_channel = mapping_dictionary[args.variable_to_plot][index]

    predictions = load_verification_data(args.path_prediction_file)
    time_steps = generate_time_steps()

    target_all = xr.open_zarr(args.path_target_file, consolidated=True)[
        args.variable_to_plot
    ].isel(time=slice(0, len(time_steps)))

    # Create the figure outside the loop
    fig, axes = plt.subplots(
        2,
        1,
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": constants.SELECTED_PROJ},
    )

    for i, _ in time_steps.items():
        target = target_all.isel(time=i)
        if constants.IS_3D[args.variable_to_plot]:
            target = target[feature_channel]

        # Reshape the data to fit the plots
        target_array = np.moveaxis(target.values, [-2, -1], [-1, -2])

        prediction_array = (
            predictions[0, i, :, feature_channel]
            .reshape(*constants.GRID_SHAPE[::-1])
            .cpu()
            .numpy()
        )

        # Plot both on the same axes array
        create_geographic_plot(
            target_array, feature_channel, "Ground Truth", axes[0]
        )
        contour_set = create_geographic_plot(
            prediction_array, feature_channel, "Prediction", axes[1]
        )

    # Add colorbar and save the figure
    cbar = fig.colorbar(
        contour_set,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        aspect=40,
    )
    cbar.ax.tick_params(labelsize=10)

    # Save plot
    directory = os.path.dirname(args.saving_path)
    plot = f"{args.saving_path}comparison_plot.png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(plot, bbox_inches="tight")
    plt.close()


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
