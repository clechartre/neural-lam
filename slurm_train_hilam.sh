#!/bin/bash -l
#SBATCH --job-name=NeurWPh
#SBATCH --mem=490G
#SBATCH --output=lightning_logs/neurwph.out
#SBATCH --error=lightning_logs/neurwph.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100-80gb
#SBATCH --account=s83

# Load necessary modules
conda activate neural-ddp

export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py \
    --dataset "straka" --val_interval 10 --epochs 200 --n_workers 16 \
    --batch_size 0 --model hi_lam --graph hierarchical
