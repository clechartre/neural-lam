#!/bin/bash -l
#SBATCH --job-name=offlineplot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=pp-short
#SBATCH --account=s83
#SBATCH --output=lightning_logs/offline_out.log
#SBATCH --error=lightning_logs/offline_err.log
#SBATCH --time=01:00:00
#SBATCH --no-requeue

conda activate neural-lam
ulimit -c 0
export OMP_NUM_THREADS=16

srun -ul python offline.py --path_target_file "/users/clechart/neural-lam/data/cosmo/samples/predict/data.zarr" \
    --path_prediction_file "/users/clechart/neural-lam/templates/predictions.npy" \
    --saving_path "/users/clechart/neural-lam/figures/" \
    --variable_to_plot "QV"
