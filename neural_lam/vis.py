import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_lam import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, title=None, step_length=1):
    """
    Plot a line plot of errors of a single variable at different predictions horizons
    errors: (pred_steps,)
    """
    errors_np = errors.cpu().numpy()  # (pred_steps, d_f)
    pred_steps = int(errors_np.shape[0])
    fig, ax = plt.subplots(figsize=constants.fig_size)

    ax.plot(errors_np, color="darkred")

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticks(pred_hor_i[::10])
    ax.set_xticklabels(pred_hor_h[::10], size=label_size)
    ax.set_xlabel("Lead time (10s)", size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, title=None, vrange=None):
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

    fig, axes = plt.subplots(2, 1, figsize=constants.fig_size)

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        data_grid = np.transpose(data.reshape(*constants.grid_shape).cpu().numpy())
        data_grid = np.flipud(data_grid)  # Flip the data along the vertical axis
        contour_set = ax.contourf(
            data_grid,
            cmap="plasma",
            levels=np.linspace(
                vmin,
                vmax,
                num=100))

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
    Error has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    fig_size = constants.fig_size
    fig_size = list(fig_size)
    fig_size[1] /= 2
    fig_size = tuple(fig_size)
    fig, ax = plt.subplots(figsize=constants.fig_size)

    error_grid = np.transpose(error.reshape(*constants.grid_shape).cpu().numpy())
    error_grid = np.flipud(error_grid)  # Flip the data along the vertical axis
    contour_set = ax.contourf(
        error_grid,
        cmap="plasma",
        levels=np.linspace(
            vmin,
            vmax,
            num=100))

    # Ticks and labels
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
