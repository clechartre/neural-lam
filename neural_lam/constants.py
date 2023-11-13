import numpy as np

data_config = {
    "data_path": "/scratch/mch/sadamov/pyprojects_data/gwen/data/experiments/",
    "filename_regex": "atmcirc-straka_93_(.*)DOM01_ML_20080801T000000Z.nc",
    "zarr_path": "/scratch/mch/sadamov/pyprojects_data/gwen/data/data_combined.zarr",
    "zlib_compression_level": 1
}

wandb_project = "straka"

# Full names
param_names = [
    'Potential Temperature',
]
# Short names
param_names_short = [
    'theta_v',
]

# Units
param_units = [
    'K',
]
# Projection and grid
grid_shape = (2632, 64)  # original (x, y)

# Zoom for graph plotting
zoom_limit = 128

# Time step prediction during training / prediction (eval)
init_time = 20 # 10s steps (after bubble drop)
train_horizon = 3  # 10s steps (t-1 + t -> t+1)
eval_horizon = 80  # 10s steps (autoregressive)

# Plotting
fig_size = (15, 10)
example_file = "data/straka/samples/test/data_test.zarr"

# Log prediction error for these time steps forward
val_step_log_errors = np.arange(1, eval_horizon - 1, 10)
metrics_initialized = False

# Some constants useful for sub-classes
batch_static_feature_dim = 0
grid_forcing_dim = 0
grid_state_dim = 1
