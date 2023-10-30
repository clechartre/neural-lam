#!/bin/bash
#SBATCH --job-name=NeurWPhi
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --output=logs/neurwp_hilam.out
#SBATCH --error=logs/neurwp_hilam.err
#SBATCH --exclusive
#SBATCH --mem=490G

# Load necessary modules
source ${SCRATCH}/miniforge3/etc/profile.d/conda.sh
conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=4

NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

# Run the script with torchrun
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS train_model.py \
    --dataset "cosmo" --val_interval 5 --epochs 200 --n_workers 16 \
    --batch_size 4 --model hi_lam --graph hierarchical
