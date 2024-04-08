#!/bin/bash -l
#SBATCH --job-name=PlotStd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=pp-short
#SBATCH --account=s83
#SBATCH --output=lightning_logs/standalone_plot_out.log
#SBATCH --error=lightning_logs/standalone_plot_err.log
#SBATCH --time=00:30:00
#SBATCH --no-requeue

ulimit -c 0
export OMP_NUM_THREADS=16


# Load necessary modules
conda activate neural-lam


srun -ul python standlone_plot_pred.py