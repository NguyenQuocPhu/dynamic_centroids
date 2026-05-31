#!/bin/bash

DATASETS=(
    "./data/formatted_DangerousQA.json"
    "./data/formatted_harmfulQA.json"
)



for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"
    python -m rainbowplus.rainbowplus \
        --config_file configs/base.yml \
        --dataset "$DATASET" \
done
