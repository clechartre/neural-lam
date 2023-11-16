#!/bin/bash -l
#SBATCH --job-name=NeurWPe
#SBATCH --mem=490G
#SBATCH --output=lightning_logs/neurwp_eval.out
#SBATCH --error=lightning_logs/neurwp_eval.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --account=s83

conda activate neural-ddp

# Set OMP_NUM_THREADS to a value greater than 1
export OMP_NUM_THREADS=16

# Run the script with torchrun
srun -ul --gpus-per-task=1 python train_model.py --n_example_pred 10 \
    --load "saved_models/graph_lam-4x64-11_15_22_38_47/last.ckpt" \
    --dataset "straka" --eval="test" --n_workers 8 --batch_size 1 --model "graph_lam"
