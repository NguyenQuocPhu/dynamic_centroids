#!/bin/bash

DATASETS=(
    "./data/harmfulQA.json"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    python -m rainbowplus.rainbowplus \
        --config_file configs/base-opensource.yml \
        --dataset "$DATASET" \
        --shuffle
done