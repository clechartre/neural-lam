#!/bin/bash

# Define the parameters and levels
params=("theta_v")
member=("1" "2" "3" "4" "5" "6" "7" "8" "9")

# Get the output directory from the first script argument
output_dir=$1

# Loop over all combinations
for param in "${params[@]}"; do
    for member in "${member[@]}"; do
        # Generate the GIF
        convert -delay 100 -loop 0 ${output_dir}/${param}_m_${member}_t* ${output_dir}/${param}_m_${member}.gif
    done
done
