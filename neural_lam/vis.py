import cartopy.feature as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from neural_lam import constants, utils
from neural_lam.rotate_grid import unrotate_latlon

plt.rc('text', usetex=False)

@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, global_mean, title=None, step_length=1):
    """
    Plot a heatmap of errors of different variables at different predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    rel_errors = errors_np / np.abs(np.expand_dims(global_mean.cpu(), axis=1))
    height = int(np.sqrt(len(constants.vertical_levels)
                         * len(constants.param_names_short)) * 2)
    fig, ax = plt.subplots(figsize=(15, height))

    ax.imshow(rel_errors, cmap="OrRd", vmin=0, vmax=1., interpolation="none",
              aspect="auto", alpha=0.8)

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha='center', va='center', usetex=False)

    # Ticks and labels
    label_size = 12
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name if name != 'RELHUM' else 'RH'} ({unit}) {f'{level:02}' if constants.is_3d[name] else ''}"
        for name, unit in zip(constants.param_names_short, constants.param_units)
        for level in (constants.vertical_levels if constants.is_3d[name] else [0])]
    y_ticklabels = sorted(y_ticklabels)
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, forecast, obs_mask, title=None, vrange=None):
    """
    Plot example prediction and ground truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange[0].cpu().item(), vrange[1].cpu().item()

    # get test data
    data_latlon = xr.open_zarr(constants.example_file).isel(time=0)
    lon, lat = unrotate_latlon(data_latlon)

    fig, axes = plt.subplots(3, 1, figsize=constants.fig_size,
                             subplot_kw={"projection": constants.selected_proj})

    # Plot pred and target
    for ax, data in zip(axes, (target, pred, forecast)): 
        data_grid = data.reshape(*constants.grid_shape[::-1]).cpu().numpy()
        contour_set = ax.contourf(
            lon,
            lat,
            data_grid,
            transform=constants.selected_proj,
            cmap="plasma",
            levels=np.linspace(
                vmin,
                vmax,
                num=100))
        ax.add_feature(cf.BORDERS, linestyle='-', edgecolor='black')
        ax.add_feature(cf.COASTLINE, linestyle='-', edgecolor='black')
        ax.gridlines(
            crs=constants.selected_proj,
            draw_labels=False,
            linewidth=0.5,
            alpha=0.5)

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    axes[2].set_title("Forecast", size=15)
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, title=None, vrange=None):
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
    data_latlon = xr.open_zarr(constants.example_file).isel(time=0)
    lon, lat = unrotate_latlon(data_latlon)

    fig, ax = plt.subplots(figsize=constants.fig_size,
                           subplot_kw={"projection": constants.selected_proj})

    error_grid = error.reshape(*constants.grid_shape[::-1]).cpu().numpy()

    contour_set = ax.contourf(
        lon,
        lat,
        error_grid,
        transform=constants.selected_proj,
        cmap="OrRd",
        levels=np.linspace(
            vmin,
            vmax,
            num=100))
    ax.add_feature(cf.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cf.COASTLINE, linestyle='-', edgecolor='black')
    ax.gridlines(
        crs=constants.selected_proj,
        draw_labels=False,
        linewidth=0.5,
        alpha=0.5)

    # Ticks and labels
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
