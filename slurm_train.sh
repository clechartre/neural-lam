#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=1
#SBATCH --job-name=NeurWP
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --mem=490G
#SBATCH --no-requeue

export PREPROCESS=true
export NORMALIZE=false

PREPROCESS=false

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    echo "Creating mesh"
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo" --plot 1
    echo "Creating grid features"
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        echo "Creating normalization weights"
        srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 32 --n_workers 8 --step_length 1
    fi
fi

ulimit -c 0
export OMP_NUM_THREADS=16

if $PREPROCESS; then
    srun -ul python tools/create_static_features.py --boundaries 60
    srun -ul python tools/create_mesh.py --dataset "cosmo"
    srun -ul python tools/create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul python tools/create_parameter_weights.py --dataset "cosmo" --batch_size 12 --n_workers 8 --step_length 1
fi

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py --dataset "cosmo" --val_interval 5 \
    --epochs 10 --n_workers 6 --batch_size 8 --subset_ds 1 --wandb_mode "offline"
