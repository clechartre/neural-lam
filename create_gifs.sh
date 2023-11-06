#!/bin/bash

# Define the parameters and levels
params=("theta_v")

# Get the output directory from the first script argument
output_dir=$1

# Loop over all combinations
for param in "${params[@]}"; do
    # Generate the GIF
    convert -delay 20 -loop 0 ${output_dir}/${param}_t_* ${output_dir}/${param}.gif
done
