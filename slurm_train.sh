#!/bin/bash -l
#SBATCH --job-name=NeurWP
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=lightning_logs/neurwp_out.log
#SBATCH --error=lightning_logs/neurwp_err.log
#SBATCH --mem=490G

PREPROCESS=false

# Load necessary modules
conda activate neural-lam

if [ ! -d "data" ]; then
    mkdir data && pushd data
    mkdir cosmo && pushd cosmo
    ln -s /scratch/mch/sadamov/pyprojects_data/neural_lam/data/cosmo/samples
    mkdir static
    popd && popd
fi

export OMP_NUM_THREADS=16

if $PREPROCESS; then
    srun -ul python tools/create_static_features.py --boundaries 60
    srun -ul python tools/create_mesh.py --dataset "cosmo"
    srun -ul python tools/create_grid_features.py --dataset "cosmo"
    # This takes multiple hours!
    srun -ul python tools/create_parameter_weights.py --dataset "cosmo" --batch_size 12 --n_workers 8 --step_length 1
fi

# Run the script with torchrun
srun -ul --gpus-per-task=1 python tools/train_model.py \
    --dataset "cosmo" --val_interval 1 --epochs 1 --n_workers 8 \
    --batch_size 12 --model "graph_lam" --graph "multiscale"
