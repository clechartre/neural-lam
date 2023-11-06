from argparse import ArgumentParser

import numpy as np
import xarray as xr

from neural_lam import constants


def main():
    parser = ArgumentParser(description='Static features arguments')
    parser.add_argument('--xdim', type=str, default="ncells",
                        help='Name of the x-dimension in the dataset (default: ncells)')
    parser.add_argument('--ydim', type=str, default="height",
                        help='Name of the x-dimension in the dataset (default: height)')
    parser.add_argument(
        '--outdir', type=str, default="data/straka/static/",
        help='Output directory for the static features (default: data/straka/static/)')
    args = parser.parse_args()

    # Open the .zarr archive
    ds = xr.open_dataset(constants.example_file)

    # Get the dimensions of the dataset
    x_dim, y_dim = ds.dims[args.xdim], ds.dims[args.ydim]

    # Create a 2D meshgrid for x and y indices
    x_grid, y_grid = np.indices((x_dim, y_dim))

    # Invert the order of x_grid
    x_grid = np.transpose(x_grid)
    y_grid = np.transpose(y_grid)

    # Stack the 2D arrays into a 3D array with x and y as the first dimension
    grid_xy = np.stack((x_grid, y_grid))

    np.save(args.outdir + 'nwp_xy.npy', grid_xy)


if __name__ == "__main__":
    main()
