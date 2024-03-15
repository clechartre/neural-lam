# Standard library
import glob
import os
from datetime import datetime, timedelta

# Third-party
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
# BUG: Import should work in interactive mode as well -> create pypi package
from neural_lam import constants, utils

# pylint: disable=W0613:unused-argument
# pylint: disable=W0201:attribute-defined-outside-init


class WeatherDataset(torch.utils.data.Dataset):
    """
    N_t = 1h
    N_x = 582
    N_y = 390
    N_grid = 582*390 = 226980
    d_features = 4(features) * 21(vertical model levels) = 84
    d_forcing = 0
    #TODO: extract incoming radiation from KENDA
    """

    def __init__(
        self,
        dataset_name,
        split="train",
        standardize=True,
        subset=False,
        batch_size=4,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join(
            "data", dataset_name, "samples", split
        )

        self.forecast_dir_path = os.path.join(
            "data", dataset_name, "samples", "test copy"
        ) if split != "forecast" else self.sample_dir_path

        self.batch_size = batch_size
        self.batch_index = 0
        self.index_within_batch = 0

        self.zarr_files = sorted(
            glob.glob(os.path.join(self.sample_dir_path, "data*.zarr"))
        )
        self.forecast_files = sorted(
            glob.glob(os.path.join(self.forecast_dir_path, "data*.zarr"))
        ) if split != "forecast" else []

        if len(self.zarr_files) == 0 and len(self.forecast_files) == 0:
            raise ValueError("No .zarr files found in directory")

        if subset:
            if constants.EVAL_DATETIME is not None and split == "test":
                eval_datetime_obj = datetime.strptime(
                    constants.EVAL_DATETIME, "%Y%m%d%H"
                )
                for i, file in enumerate(self.zarr_files):
                    file_datetime_str = file.split("/")[-1].split("_")[1][:-5]
                    file_datetime_obj = datetime.strptime(
                        file_datetime_str, "%Y%m%d%H"
                    )
                    if (
                        file_datetime_obj
                        <= eval_datetime_obj
                        < file_datetime_obj
                        + timedelta(hours=constants.CHUNK_SIZE)
                    ):
                        # Retrieve the current file and the next file if it
                        # exists
                        next_file_index = i + 1
                        if next_file_index < len(self.zarr_files):
                            self.zarr_files = [
                                file,
                                self.zarr_files[next_file_index],
                            ]
                        else:
                            self.zarr_files = [file]
                        position_within_file = int(
                            (
                                eval_datetime_obj - file_datetime_obj
                            ).total_seconds()
                            // 3600
                        )
                        self.batch_index = (
                            position_within_file // self.batch_size
                        )
                        self.index_within_batch = (
                            position_within_file % self.batch_size
                        )
                        break
            else:
                self.zarr_files = self.zarr_files[0:2] # FIXME what does this exactly do? 
            
            self.forecast_files = self.forecast_files[0:2]

            start_datetime = (
                self.zarr_files[0]
                .split("/")[-1]
                .split("_")[1]
                .replace(".zarr", "")
            )

            print("Data subset of 200 samples starts on the", start_datetime)

        self.zarr_files = self.get_3d_and_2d(self.zarr_files)
        self.forecast_files = self.get_3d_and_2d(self.forecast_files)

        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            if constants.GRID_FORCING_DIM > 0:
                self.data_mean, self.data_std, self.flux_mean, self.flux_std = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                    ds_stats["flux_mean"],
                    ds_stats["flux_std"],
                )
            else:
                self.data_mean, self.data_std = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                )

        self.random_subsample = split == "train"
        self.split = split

    @staticmethod
    def get_3d_and_2d(dataset):
        # Separate 3D and 2D variables
        variables_3d = [
            var for var in constants.PARAM_NAMES_SHORT if constants.IS_3D[var]
        ]
        variables_2d = [
            var
            for var in constants.PARAM_NAMES_SHORT
            if not constants.IS_3D[var]
        ]

        # Stack 3D variables
        datasets_3d = [
            xr.open_zarr(file, consolidated=None)[variables_3d] # Consolidate=True conflicts with the metadata file
            .sel(z_1=constants.VERTICAL_LEVELS) # FIXME this is where the issue is - reindexing data using an index that does not have unique values?
            .to_array()
            .stack(var=("variable", "z_1"))
            .transpose("time", "x_1", "y_1", "var")
            for file in dataset
        ]

        # Stack 2D variables without selecting along z_1
        datasets_2d = [
            xr.open_zarr(file, consolidated=None)[variables_2d]
            .to_array()
            # Add check to only expand_dims if 'z_1' is not already present
            .pipe(lambda ds: ds if 'z_1' in ds.dims else ds.expand_dims(z_1=[0]))
            .stack(var=("variable", "z_1"))
            .transpose("time", "x_1", "y_1", "var")
            for file in dataset
        ]

        # Combine 3D and 2D datasets
        dataset = [
            xr.concat([ds_3d, ds_2d], dim="var").sortby("var")
            for ds_3d, ds_2d in zip(datasets_3d, datasets_2d)
        ]

        return dataset



    def __len__(self):
        num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )
        total_time = len(self.zarr_files) * constants.CHUNK_SIZE - num_steps
        return total_time

    def __getitem__(self, idx):
        num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )

        # Calculate which zarr files need to be loaded
        start_file_idx = idx // constants.CHUNK_SIZE
        end_file_idx = (idx + num_steps) // constants.CHUNK_SIZE
        # Index of current slice
        idx_sample = idx % constants.CHUNK_SIZE

        # Calculate which forecast files need to be loaded 
        forecast_start_file_idx = idx // (2*constants.CHUNK_SIZE)
        forecast_end_file_idx = (idx + 2*num_steps) // (2*constants.CHUNK_SIZE)
        # Index of current slice
        forecast_idx_sample = idx % (2*constants.CHUNK_SIZE)

        sample_archive = xr.concat(
            self.zarr_datasets[start_file_idx : end_file_idx + 1], dim="time"
        )
        forecast_archive = xr.concat(
            self.forecast_files[forecast_start_file_idx : forecast_end_file_idx + 1], dim = "time" # FIXME check it's right to use forecast_files as equivalent to zarr_datasets
        )


        sample_xr = sample_archive.isel(
            time=slice(idx_sample, idx_sample + num_steps)
        )
        forecast_xr = forecast_archive.isel(
            time=slice(forecast_idx_sample, forecast_idx_sample + 2*num_steps) # FIXME maybe num steps could also be twice then - depending on the temporal scale of the dataset
        )

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)
        forecast_sample = torch.tensor(forecast_xr.values, dtype=torch.float32)

        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)
        forecast_sample = forecast_sample.flatten(1,2)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std
            forecast_sample = (forecast_sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)
        forecast_states = forecast_sample 

        return init_states, target_states, forecast_states


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        dataset_name,
        split="train",
        standardize=True,
        subset=False,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.subset = subset

    def prepare_data(self):
        # download, split, etc... called only on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test split) called on every process
        # in DDP
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                self.dataset_name,
                split="train",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                self.dataset_name,
                split="val",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                self.dataset_name,
                split="test",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset.zarr_files,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )