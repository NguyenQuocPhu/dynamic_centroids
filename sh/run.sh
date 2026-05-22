#!/bin/bash

DATASETS=(
    "./data/formatted_harmbench.json"
)

PARAMETERS=(
    "2,3"
    "1,7"
    "2,2"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    for PARAM in "${PARAMETERS[@]}"; do
        TOTAL=$(echo "$PARAM" | cut -d',' -f1)
        TREE_WIDTH=$(echo "$PARAM" | cut -d',' -f2)

        echo "Running with total: $TOTAL"
        echo "Running with tree width: $TREE_WIDTH"

        python -m rainbowplus.rainbowplus \
            --config configs/base.yml \
            --dataset "$DATASET" \
            --total "$TOTAL" \
            --tree_width "$TREE_WIDTH"
    done
done