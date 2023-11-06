"""Load weather data from NetCDF files and store it in a Zarr archive."""
# Standard library
import os
import re
from typing import List

import numcodecs

# Third-party
import xarray as xr

from neural_lam.constants import data_config


def append_or_create_zarr(data_out: xr.Dataset, config: dict) -> None:
    """Append data to an existing Zarr archive or create a new one."""
    if os.path.exists(config["zarr_path"]):
        data_out.to_zarr(
            store=config["zarr_path"],
            mode="a",
            consolidated=True,
            append_dim="member",
        )
    else:
        data_out.to_zarr(
            config["zarr_path"],
            mode="w",
            consolidated=True,
        )


def load_data(config: dict) -> None:
    """Load weather data from NetCDF files and store it in a Zarr archive.

    The data is assumed to be in a specific directory structure and file naming
    convention, which is checked using regular expressions. The loaded data is chunked
    along the "member" and "time" dimensions for efficient storage in the Zarr archive.
    If the Zarr archive already exists, new data is appended to it. Otherwise, a new
    Zarr archive is created.

    Args:
        None

    Returns:
        None

    """
    for folder in config["folders"]:
        if folder.startswith("atmcirc-straka_93_"):
            file_path: str = os.path.join(config["data_path"], folder)
            files: List[str] = os.listdir(file_path)
            for file in files:
                try:
                    match = config["filename_pattern"].match(file)
                    if not match:
                        continue

                    data: xr.Dataset = xr.open_dataset(
                        os.path.join(file_path, file), engine="netcdf4"
                    )

                    # Specify the encoding for theta_v
                    if "theta_v" in data:
                        data["theta_v"].encoding = {"compressor": config["compressor"]}

                    data = data.assign_coords(member=match.group(1))
                    data = data.expand_dims({"member": 1})
                    data = data.chunk(
                        chunks={
                            "time": 3,
                            "member": 1,
                            "height": -1,
                            "height_2": -1,
                            "height_3": -1,
                        }
                    )
                    append_or_create_zarr(data, config)
                except (FileNotFoundError, OSError) as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    data_config.update(
        {
            "folders": os.listdir(data_config["data_path"]),
            "filename_pattern": re.compile(data_config["filename_regex"]),
            "compressor": numcodecs.Zlib(level=data_config["zlib_compression_level"]),
        }
    )
    load_data(data_config)
