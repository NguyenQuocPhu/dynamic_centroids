#!/bin/bash

DATASETS=(
    "./data/AdversarialQA.json"
    "./data/BeaverTails.json"
    "./data/harmfulQA.json"
    "./data/CategoricalHarmfulQA.json"
    "./data/DangerousQA.json"
    "./data/do-not-answer.json"
    "./data/harmbench.json"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    python -m rainbowplus.rainbowplus \
        --config_file configs/base.yml \
        --dataset "$DATASET" \
done