#!/bin/bash

DATASETS=(
    "./data/formatted_do-not-answer.json"
    "./data/formatted_harmbench.json"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    python -m rainbowplus.rainbowplus \
        --config_file configs/base.yml \
        --dataset "$DATASET" 
done