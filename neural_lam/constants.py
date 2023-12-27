import numpy as np
from cartopy import crs as ccrs

wandb_project = "neural-lam"

# Full names
param_names = [
    'Temperature',
    'Zonal wind component',
    'Meridional wind component',
    'Relative humidity',
    # 'Geopotential',
    'Pressure at Mean Sea Level',
    # 'Reference Pressure',
    'Pressure Perturbation',
    'Surface Pressure',
    # 'Total Precipitation',
    'Total Water Vapor content',
    '2-meter Temperature',
    '10-meter Zonal wind speed',
    '10-meter Meridional wind speed',
]

# Short names
param_names_short = [
    'T',
    'U',
    'V',
    'RELHUM',
    # 'FI',
    'PMSL',
    # 'P0FL',
    'PP',
    'PS',
    # 'TOT_PREC',
    'TQV',
    'T_2M',
    'U_10M',
    'V_10M',
]

# Units
param_units = [
    'K',
    'm/s',
    'm/s',
    'Perc.',
    # '$m^2/s^2$',
    'Pa',
    # 'Pa',
    'Pa',
    'Pa',
    # 'mm',
    '$kg/m^2$',
    'K',
    'm/s',
    'm/s',
]

# Parameter weights
param_weights = {
    'T': 1,
    'U': 1,
    'V': 1,
    'RELHUM': 1,
    # 'FI': 1,
    'PMSL': 1,
    # 'P0FL': 1,
    'PP': 1,
    'PS': 1,
    # 'TOT_PREC': 1,
    'TQV': 1,
    'T_2M': 1,
    'U_10M': 1,
    'V_10M': 1,
}

# Vertical levels
vertical_levels = [
    1, 5, 13, 22, 38, 41, 60
]

is_3d = {
    'T': 1,
    'U': 1,
    'V': 1,
    'RELHUM': 1,
    # 'FI': 1,
    'PMSL': 0,
    # 'P0FL': 1,
    'PP': 1,
    'PS': 0,
    # 'TOT_PREC': 0,
    'TQV': 0,
    'T_2M': 0,
    'U_10M': 0,
    'V_10M': 0,
}

# Vertical level weights
level_weights = {
    1: 1,
    5: 1,
    13: 1,
    22: 1,
    38: 1,
    41: 1,
    60: 1,
}

# Projection and grid
grid_shape = (390, 582)  # (y, x)

# Time step prediction during training / prediction (eval)
train_horizon = 6  # hours (t-1 + t -> t+1)
eval_horizon = 25  # hours (autoregressive)

# Log prediction error for these time steps forward
val_step_log_errors = np.arange(1, eval_horizon - 1)
metrics_initialized = False

# Plotting
fig_size = (15, 10)
example_file = "data/cosmo/samples/train/data_2015112800.zarr"
chunk_size = 100
eval_sample = 0
store_example_data = False
cosmo_proj = ccrs.PlateCarree()
selected_proj = cosmo_proj
pollon = -170.0
pollat = 43.0

# Some constants useful for sub-classes
batch_static_feature_dim = 0
grid_forcing_dim = 0
