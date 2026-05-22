#!/bin/bash

DATASETS=(
    "./data/formatted_harmbench.json"
)

TOTALS=(
    2
    3
    4
)
TREE_WIDTHS=(
    2
)
for DATASET in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATASET"

    for TOTAL in "${TOTALS[@]}"; do
        echo "Running with total: $TOTAL"
        for TREE_WIDTH in "${TREE_WIDTHS[@]}"; do
            echo "Running with tree width: $TREE_WIDTH"
            python -m rainbowplus.rainbowplus --config configs/base.yml --dataset "$DATASET" --total "$TOTAL" --tree_width "$TREE_WIDTH"
        done
    done
done
