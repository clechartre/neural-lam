import os
from argparse import ArgumentParser

import numpy as np
import torch


def main():
    parser = ArgumentParser(description='Training arguments')
    parser.add_argument('--dataset', type=str, default="meps_example",
                        help='Dataset to compute weights for (default: meps_example)')
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # -- Static grid node features --
    grid_xy = torch.tensor(np.load(os.path.join(static_dir_path, "nwp_xy.npy")
                                   ))  # (2, N_x, N_y)
    grid_xy = grid_xy.flatten(1, 2).T  # (N_grid, 2)
    pos_max = torch.max(torch.abs(grid_xy))
    grid_xy = grid_xy / pos_max  # Divide by maximum coordinate

    # Concatenate grid features
    grid_features = grid_xy # (N_grid, 1)

    torch.save(grid_features, os.path.join(
        static_dir_path, "grid_features.pt"))


if __name__ == "__main__":
    main()
