#!/bin/bash

DATASETS=(
    "./data/formatted_AdversarialQA.json"
    "./data/formatted_BeaverTails.json"
    "./data/formatted_CategoricalHarmfulQA.json"
    "./data/formatted_DangerousQA.json"
    "./data/formatted_do-not-answer.json"
    "./data/formatted_harmbench.json"
    "./data/formatted_harmfulQA.json"
)

PARAMETERS=(
    "3,2"
    "7,1"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    for PARAM in "${PARAMETERS[@]}"; do
        TOTAL=$(echo "$PARAM" | cut -d',' -f1)
        TREE_WIDTH=$(echo "$PARAM" | cut -d',' -f2)

        echo "Running with total: $TOTAL"
        echo "Running with tree width: $TREE_WIDTH"

        python -m rainbowplus.rainbowplus \
            --config_file configs/base.yml \
            --dataset "$DATASET" \
            --total "$TOTAL" \
            --tree_width "$TREE_WIDTH"
    done
done
