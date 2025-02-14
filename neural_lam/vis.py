# Third-party
import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# First-party
from neural_lam import constants, utils
from neural_lam.rotate_grid import unrotate_latlon


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
            f"{f'{level:02}' if constants.IS_3D[name] else ''}"
        )
        for name, unit in zip(
            constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
        )
        for level in (
            constants.VERTICAL_LEVELS if constants.IS_3D[name] else [0]
        )
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
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange[0].cpu().item(), vrange[1].cpu().item()

    # get test data
    data_latlon = xr.open_zarr(constants.EXAMPLE_FILE).isel(time=0)
    lon, lat = unrotate_latlon(data_latlon)

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
    lon, lat = unrotate_latlon(data_latlon)

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
