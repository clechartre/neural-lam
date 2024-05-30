# Standard library
import os
import subprocess
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# First-party
from neural_lam import config
from neural_lam.weather_dataset import WeatherDataset


def get_rank():
    """Get the rank of the current process in the distributed group."""
    return int(os.environ["SLURM_PROCID"])


def get_world_size():
    """Get the number of processes in the distributed group."""
    return int(os.environ["SLURM_NTASKS"])


def setup(rank, world_size):  # pylint: disable=redefined-outer-name
    """Initialize the distributed group."""
    try:
        master_node = (
            subprocess.check_output(
                "scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1",
                shell=True,
            )
            .strip()
            .decode("utf-8")
        )
    except Exception as e:
        print(f"Error getting master node IP: {e}")
        raise
    master_port = "12355"
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = master_port
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Destroy the distributed group."""
    dist.destroy_process_group()


def main(rank, world_size):  # pylint: disable=redefined-outer-name
    """Compute the mean and standard deviation of the input data."""
    setup(rank, world_size)
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)
    static_dir_path = os.path.join("data", config_loader.dataset.name, "static")

    # Create parameter weights based on height
    # based on fig A.1 in graph cast paper
    w_dict = {
        "2": 1.0,
        "0": 0.1,
        "65": 0.065,
        "1000": 0.1,
        "850": 0.05,
        "500": 0.03,
    }
    w_list = np.array(
        [
            w_dict[par.split("_")[-2]]
            for par in config_loader.dataset.var_longnames
        ]
    )
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"),
        w_list.astype("float32"),
    )
    static_dir_path = os.path.join("data", args.dataset, "static")

    # Load dataset without any subsampling
    ds = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=False,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    data_module.setup(stage="fit")

    train_sampler = DistributedSampler(
        data_module.train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.n_workers,
    )

    if rank == 0:
        w_list = [
            pw * lw
            for var_name, pw in zip(
                constants.PARAM_NAMES_SHORT, constants.PARAM_WEIGHTS.values()
            )
            for lw in (
                constants.LEVEL_WEIGHTS.values()
                if constants.IS_3D[var_name]
                else [1]
            )
        ]
        np.save(
            os.path.join(static_dir_path, "parameter_weights.npy"),
            np.array(w_list, dtype="float32"),
        )

    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, forcing_batch in tqdm(loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

        # Flux at 1st windowed position is index 1 in forcing
        flux_batch = forcing_batch[:, :, :, 1]
        flux_means.append(torch.mean(flux_batch))  # (,)
        flux_squares.append(torch.mean(flux_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    flux_mean = torch.mean(torch.stack(flux_means))  # (,)
    flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
    flux_stats = torch.stack((flux_mean, flux_std))

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))
    torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=True,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    data_module.setup(stage="fit")

    train_sampler = DistributedSampler(
        data_module.train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.n_workers,
    )

    # Compute mean and std-dev of one-step differences
    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _ in tqdm(train_loader, disable=rank != 0):
        batch = torch.cat((init_batch, target_batch), dim=1).to(device)
        diffs = batch[:, 1:] - batch[:, :-1]
        diff_means.append(torch.mean(diffs, dim=(1, 2)))
        diff_squares.append(torch.mean(diffs**2, dim=(1, 2)))

    dist.barrier()

    diff_means_gathered = [None] * world_size
    diff_squares_gathered = [None] * world_size
    dist.all_gather_object(diff_means_gathered, torch.cat(diff_means, dim=0))
    dist.all_gather_object(
        diff_squares_gathered, torch.cat(diff_squares, dim=0)
    )

    if rank == 0:
        diff_means_all = torch.cat(diff_means_gathered, dim=0)
        diff_squares_all = torch.cat(diff_squares_gathered, dim=0)
        diff_mean = torch.mean(diff_means_all, dim=0)
        diff_second_moment = torch.mean(diff_squares_all, dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)
        torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
        torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))

    cleanup()


if __name__ == "__main__":
    rank = get_rank()
    world_size = get_world_size()
    main(rank, world_size)
