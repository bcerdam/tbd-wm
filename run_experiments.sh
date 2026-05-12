#!/bin/bash

# Define the number of runs
N=5
# Define the base output directory
BASE_DIR="output/experiments"

for ((i=1; i<=N; i++))
do
    echo "========================================="
    echo "Starting Run $i of $N"
    echo "========================================="
    
    # Define a unique directory for this specific run
    RUN_DIR="${BASE_DIR}/run_${i}"
    
    # Execute the training script, passing only the run directory
    python -u train.py \
        --run_dir "$RUN_DIR"
        
    echo "Run $i completed!"
done

echo "All $N runs finished successfully."
