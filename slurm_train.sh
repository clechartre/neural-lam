#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --account=s83
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log

export PREPROCESS=false
export NORMALIZE=false
export DATASET="cosmo"
export MODEL="hi_lam"

# Load necessary modules
conda activate neural-lam

if [ "$PREPROCESS" = true ]; then
    echo "Create static features"
    python create_static_features.py --boundaries 60 --dataset $DATASET
    if [ "$MODEL" = "hi_lam" ]; then
        echo "Creating hierarchical mesh"
        python create_mesh.py --dataset $DATASET --plot 1 --graph hierarchical --levels 4 --hierarchical 1
    else
        echo "Creating flat mesh"
        python create_mesh.py --dataset $DATASET --plot 1
    fi
    echo "Creating grid features"
    python create_grid_features.py --dataset $DATASET
    if [ "$NORMALIZE" = true ]; then
        # This takes multiple hours!
        echo "Creating normalization weights"
        sbatch slurm_param.sh
    fi
fi

echo "Training model"
if [ "$MODEL" = "hi_lam" ]; then
    srun -ul python train_model.py --dataset $DATASET --val_interval 20 --epochs 20 --n_workers 8 \
        --batch_size 12 --subset_ds 0 --model hi_lam --graph hierarchical
else
    srun -ul python train_model.py --dataset $DATASET --val_interval 20 --epochs 40 --n_workers 8 --batch_size 12 --subset_ds 0
fi
