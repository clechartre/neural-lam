"""Functions to import and preprocess weather data."""

# Standard library
import argparse

import numcodecs

# Third-party
import numpy as np
import xarray as xr
from pyprojroot import here

from neural_lam.constants import data_config


def main():
    parser = argparse.ArgumentParser(
        description="Import and preprocess weather data."
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--test_size', type=float, default=5)
    args = parser.parse_args()
    # Data Import
    try:
        data_zarr = xr.open_zarr(
            str(here()) + "/data/straka/samples/data_combined.zarr", consolidated=True
        )
        data_theta = (
            data_zarr["theta_v"]
            .sel(ncells=slice(2632, None))
            .transpose("time", "member", "height", "ncells")
        )

        # # Check for missing data
        # if np.isnan(data_theta.to_numpy()).any():

        #     # Interpolate missing data
        #     assert (
        #         "time" in data_theta.dims
        #     ), "The dimension 'time' does not exist in the dataset."
        #     data_theta = data_theta.interpolate_na(
        #         dim="time", method="linear", fill_value="extrapolate"
        #     )

    except Exception as e:

        raise ValueError(
            "Failed to import data. Please check the input data and parameters."
        ) from e

    # Split data into training and testing sets
    try:
        # Generate a random selection of members for the test set
        np.random.seed(args.seed)
        # Randomly sample members for the test set
        test_members = np.random.choice(data_theta.member, size=int(args.test_size), replace=False)
        data_test = data_theta.sel(member=test_members)
        data_train = data_theta.drop_sel(member=test_members)

    except Exception as e:

        raise ValueError(
            "Failed to split data. Please check the input data and parameters."
        ) from e

    # Chunk and compress the normalized data and save to zarr files
    try:
        data_train.chunk(
            chunks={
                "time": 3,
                "member": 1,
                "height": -1,
                "ncells": -1,
            }
        ).to_zarr(
            str(here()) + "/data/straka/samples/train/data_train.zarr",
            encoding={"theta_v": {"compressor": numcodecs.Zlib(level=data_config["zlib_compression_level"]),
                                  }},
            mode="w",
        )

        data_test.chunk(
            chunks={
                "time": 3,
                "member": 1,
                "height": -1,
                "ncells": -1,
            }
        ).to_zarr(
            str(here()) + "/data/straka/samples/test/data_test.zarr",
            encoding={"theta_v": {"compressor": numcodecs.Zlib(level=data_config["zlib_compression_level"]),
                                  }},
            mode="w",
        )

    except Exception as e:

        raise ValueError(
            "Failed to chunk and compress data."
            " Please check the input data and parameters."
        ) from e


if __name__ == "__main__":
    main()
