#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --mem=490G
#SBATCH --no-requeue

export PREPROCESS=false

# Load necessary modules
conda activate neural-ddp

if [ "$PREPROCESS" = true ]; then
    srun -ul -N1 -n1 python create_static_features.py --boundaries 60
    srun -ul -N1 -n1 python create_mesh.py --dataset "cosmo" --plot 1
    srun -ul -N1 -n1 python create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul -N1 -n1 python create_parameter_weights.py --dataset "cosmo" --batch_size 32 --n_workers 8 --step_length 1
fi

ulimit -c 0
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "cosmo" --val_interval 20 --epochs 140 --n_workers 4 --batch_size 3 \
    --load wandb/run-20240105_085036-ie0v7gzf/files/latest-v1.ckpt --resume_opt_sched 0
