# Third-party
import os
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


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(title=None, vrange=None, save_path=None):
    """
    Plot example prediction, forecast, and ground truth.
    Each has shape (N_grid,)
    """

    # Load my prediction dataset for plotting
    predictions_data_module = WeatherDataModule(
            "cosmo",
            split="pred",
            standardize=False,
            subset=False,
            batch_size=6,
            num_workers=2
        )
    predictions_data_module.setup(stage='predictions_standalone')
    predictions_loader = predictions_data_module.predictions_dataloader() 
    for predictions_batch in predictions_loader:
        predictions = predictions_batch[0]  # tensor 
        break 

    # get test data
    data_latlon = xr.open_zarr(constants.EXAMPLE_FILE).isel(time=0)
    lon, lat = unrotate_latlon(data_latlon)

    # Get common scale for values
    bb =  predictions[0,:,:,24]
    bb_array = np.array(bb)
    if vrange is None:
        vmin = bb_array.min()
        vmax = bb_array.max()
    else:
        vmin, vmax = float(vrange[0].cpu().item()), float(vrange[1].cpu().item())


    # Plot
    for i in range(23):

        aa = predictions[0,i,:,24].reshape(*constants.GRID_SHAPE[::-1]).cpu().numpy()
        data_array = np.array(aa)

        fig, axes = plt.subplots(
            1,
            1,
            figsize=constants.FIG_SIZE,
            subplot_kw={"projection": constants.SELECTED_PROJ},
        )

        contour_set = axes.contourf(
            lon,
            lat,
            data_array,
            transform=constants.SELECTED_PROJ,
            cmap="plasma",
            levels=np.linspace(vmin, vmax, num=100),
        )
        axes.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
        axes.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
        axes.gridlines(
            crs=constants.SELECTED_PROJ,
            draw_labels=False,
            linewidth=0.5,
            alpha=0.5,
        )

        # Ticks and labels
        axes.set_title("Predictions from numpy array", size = 15)
        cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
        cbar.ax.tick_params(labelsize=10)

        if title:
            fig.suptitle(title, size=20)

        # Save the plot! 
        if save_path:
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path + f"{i}.png", bbox_inches='tight')

    return fig

if __name__ == "__main__":
    plot_prediction(save_path="/users/clechart/clechart/neural-lam/lightning_logs/standalone_")