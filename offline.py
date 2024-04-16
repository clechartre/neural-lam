# Standard library
import os

# Third-party
import cartopy.feature as cf
import fsspec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# First-party
from neural_lam import constants, utils
from neural_lam.rotate_grid import unrotate_latlon
from neural_lam.weather_dataset import WeatherDataModule


def main():
    # Load data from Zarr archive using xarray
    # ds = xr.open_zarr(
    #     fsspec.get_mapper(
    #         "/users/clechart/clechart/neural-lam/data/offline/data_2020011017.zarr",
    #         anon=True,
    #     ),
    #     consolidated=True,
    # ).isel(time=1)
    # tqv = ds["TQV"]

    print("cap")
    z_1 = constants.VERTICAL_LEVELS
    newme = (
        xr.open_zarr(
            "/users/clechart/clechart/neural-lam/data/offline/data_2020011017.zarr",
            consolidated=True,
        )["TQV"]
        .isel(time=0)
        .transpose("x_1", "y_1")
    )

    # Define paths for prediction and saving output
    path_prediction_file = "/users/clechart/clechart/neural-lam/wandb/run-20240411_140635-cux0r96n/files/results/inference/prediction_0.npy"
    save_path = "/scratch/mch/tobwi/sbox/neural-lam/offline"

    # Verify inference with the target dataset
    verify_inference(
        newme,
        file_path=path_prediction_file,
        save_path=save_path,
        feature_channel=24,
    )


# Verification function to plot predictions and target data
@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def verify_inference(
    target, file_path: str, save_path: str, feature_channel: int, vrange=None
):
    # Load inference dataset
    predictions_data_module = WeatherDataModule(
        "cosmo_old",
        path_verif_file=file_path,
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
    lon, lat = unrotate_latlon(target)

    # Convert predictions to NumPy array
    prediction_array = (
        predictions[0, 1, :, 24]
        .reshape(*constants.GRID_SHAPE[::-1])
        .cpu()
        .numpy()
    )

    # Convert target data to NumPy array
    target_tensor = torch.tensor(target.values)
    target_array = target_tensor.reshape(*constants.GRID_SHAPE[::-1])
    target_feature_array = np.array(target_array)
    if vrange is None:
        vmin = target_feature_array.min()
        vmax = target_feature_array.max()
    else:
        vmin, vmax = float(vrange[0].cpu().item()), float(
            vrange[1].cpu().item()
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

    # Add colorbar and titles
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(
        os.path.join(save_path, "plot_offline.png"), bbox_inches="tight"
    )
    plt.close(fig)


# Entry point for script
if __name__ == "__main__":
    main()
