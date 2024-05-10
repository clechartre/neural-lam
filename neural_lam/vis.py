# Standard library

# Standard library
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Third-party
import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

# First-party
from neural_lam import constants, utils
from neural_lam.weather_dataset import WeatherDataModule


@dataclass
class Variable:
    """Variables and different channels in batch.
    """
    short_name: str
    channels: List[int]


@dataclass
class VariableCollection:
    """Variables and their associated channels in batch.
    """
    def __init__(self) -> None:
        self.all_products: List[Variable] = []

    def add_product(self, variable: Variable):
        if variable.short_name in self.all_products:
            raise ValueError(f"Variable {variable.short_name} already exists.")
        self.all_products.append(variable)

    def get_variable_from_shortname(self, short_name: str) -> Variable:
        for product in self.all_products:
            if product.short_name == short_name:
                return product
        raise KeyError("Variable not found")

    def get_variable_from_channel(self, channel: int) -> Variable:
        for variable in self.all_products.values():
            if channel in variable.channels:
                return variable
        raise KeyError(f"No variable found with channel {channel}")


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, global_mean, step_length=1, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    errors_norm = errors_np / np.abs(np.expand_dims(global_mean.cpu(), axis=1))
    height = int(
        np.sqrt(
            len(constants.VERTICAL_LEVELS) * len(constants.PARAM_NAMES_SHORT)
        )
        * 2
    )
    fig, ax = plt.subplots(figsize=(15, height))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        (
            f"{name if name != 'RELHUM' else 'RH'} ({unit}) "
            f"{f'{z:02}' if constants.IS_3D[name] else ''}"
        )
        for name, unit in zip(
            constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
        )
        for z in (constants.VERTICAL_LEVELS if constants.IS_3D[name] else [0])
    ]
    y_ticklabels = sorted(y_ticklabels)
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, title=None, vrange=None):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    mapping_dictionary = precompute_variable_indices()
    all_vars = VariableCollection()
    for var_name, channels in mapping_dictionary.items():
        all_vars.add_product(Variable(var_name, channels))

    try:
        variable = all_vars.get_variable_from_shortname(
            constants.EVAL_PLOT_VARS[0]
        )
        feature_channel = variable.channels[
            0
        ]  # Assuming we use the first channel
    except KeyError:
        raise ValueError(
            f"No variable found for name {constants.EVAL_PLOT_VARS[0]}"
        )

    lon, lat, vmin, vmax = set_plot_dimensions(feature_channel, vrange)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": constants.SELECTED_PROJ},
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        data_grid = data.reshape(*constants.GRID_SHAPE[::-1]).cpu().numpy()
        contour_set = ax.contourf(
            lon,
            lat,
            data_grid,
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

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange[0].cpu().item(), vrange[1].cpu().item()

    # get test data
    data_latlon = xr.open_zarr(constants.EXAMPLE_FILE).isel(time=0)
    lon, lat = data_latlon.lon.values.T, data_latlon.lat.values.T

    fig, ax = plt.subplots(
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": constants.SELECTED_PROJ},
    )

    error_grid = error.reshape(*constants.GRID_SHAPE[::-1]).cpu().numpy()

    contour_set = ax.contourf(
        lon,
        lat,
        error_grid,
        transform=constants.SELECTED_PROJ,
        cmap="OrRd",
        levels=np.linspace(vmin, vmax, num=100),
    )
    ax.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
    ax.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
    ax.gridlines(
        crs=constants.SELECTED_PROJ, draw_labels=False, linewidth=0.5, alpha=0.5
    )

    # Ticks and labels
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def verify_inference(file_path: str, save_path: str, feature_channel: int):
    """
    Plot example prediction, verification, and ground truth.
    Each has shape (N_grid,)
    """
    predictions = load_verification_data(file_path)

    if not 0 <= feature_channel < predictions.shape[-1]:
        raise ValueError(
            f"feature_channel must be between 0 and "
            f"{predictions.shape[-1] - 1}, inclusive."
        )

    # Prepare the plotting environment
    fig, axes = plt.subplots(
        1,
        1,
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": constants.SELECTED_PROJ},
    )

    for i in tqdm(
        range(constants.EVAL_HORIZON - 2), desc="Plotting predictions"
    ):
        feature_array = (
            predictions[0, i, :, feature_channel]
            .reshape(*constants.GRID_SHAPE[::-1])
            .cpu()
            .numpy()
        )
        title = (
            "Predictions from model inference: "
            f"Feature channel {feature_channel}, time step {i}"
        )
        contour_set = create_geographic_plot(
            feature_array, feature_channel, title, axes
        )

    # Add colorbar to the figure
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=40)
    cbar.ax.tick_params(labelsize=10)

    # Save the plot
    directory = os.path.dirname(save_path)
    plot_filename = f"{save_path}verification_plot.png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close(fig)


# Load the inference dataset for plotting
def load_verification_data(file_path: str) -> torch.Tensor:
    """Load data in memory as a WeatherDataModule.

    Args:
        file_path (str): path to file containing data

    Returns:
        torch.Tensor: A tensor containing the predictions of the first batch.
    """
    predictions_data_module = WeatherDataModule(
        "cosmo",
        path_verif_file=file_path,
        standardize=False,
        subset=False,
        batch_size=6,
        num_workers=2,
    )
    predictions_data_module.setup(stage="verif")
    predictions_loader = predictions_data_module.verif_dataloader()
    for predictions_batch in predictions_loader:
        predictions = predictions_batch[0]  # tensor
        break
    return predictions


def set_plot_dimensions(
    feature_channel: int,
    vrange: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """_summary_

    Args:
        feature_channel (int):the level corresponding to the
        variable and vertical level of interest
          in the prediction array
        vrange (Tuple[torch.Tensor, torch.Tensor], optional): A tuple containing
          minimum and maximum values as tensors. Defaults to None.

    Returns:
         Tuple[np.ndarray, np.ndarray, float, float]: A tuple
         containing two numpy arrays
         for longitude and latitude, and two
         floats for minimum and maximum values.
    """
    # get test data
    data_latlon = xr.open_zarr(constants.EXAMPLE_FILE).isel(time=0)
    lon, lat = data_latlon.lon.values.T, data_latlon.lat.values.T

    mapping_dictionary = precompute_variable_indices()
    keys_with_feature_channel = [
        key
        for key, values in mapping_dictionary.items()
        if feature_channel in values
    ]

    if vrange is None:
        vmin = data_latlon[keys_with_feature_channel[0]].min().values.item()
        vmax = data_latlon[keys_with_feature_channel[0]].max().values.item()
    else:
        # Convert tensor to float, assuming vrange is already properly formatted
        vmin, vmax = float(vrange[0].cpu().item()), float(
            vrange[1].cpu().item()
        )

    return lon, lat, vmin, vmax


def create_geographic_plot(data_array, feature_channel, title, ax):
    """
    Create a geographic plot with contour fill, coastlines,
      borders, and gridlines, and return the axes.

    Parameters:
        data_array (array): The data values to contour.
        feature_channel (int): The index of the feature channel being plotted.
        i (int): The index of the time step being plotted.
        title (str): Title for the plot.
        ax (matplotlib.axes): Axes object to plot on.
    """
    lon, lat, vmin, vmax = set_plot_dimensions(feature_channel, vrange=None)

    contour_set = ax.contourf(
        lon,
        lat,
        data_array,
        transform=constants.SELECTED_PROJ,
        cmap="plasma",
        levels=np.linspace(vmin, vmax, num=100),
    )
    ax.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
    ax.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
    ax.gridlines(
        crs=constants.SELECTED_PROJ, draw_labels=False, linewidth=0.5, alpha=0.5
    )
    ax.set_title(title, size=15)

    return contour_set


def precompute_variable_indices() -> dict:
    """
    Precompute indices for each variable in the input tensor.

    Returns:
        variable_indices (dict): a dictionary with the variables and
        their associated indices
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
