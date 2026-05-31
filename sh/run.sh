#!/bin/bash

DATASETS=(
    "./data/formatted_DangerousQA.json"
    "./data/formatted_harmfulQA.json"
)

DATASETS=(
    "./data/formatted_AdversarialQA.json"
    "./data/formatted_BeaverTails.json"
    "./data/formatted_do-not-answer.json"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"
    python -m rainbowplus.rainbowplus \
        --config_file configs/base.yml \
        --dataset "$DATASET" 
done
