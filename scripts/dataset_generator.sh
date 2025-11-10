#!/bin/bash

# === Configuration ===
# Automatically find the base directory (where this script lives)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${BASE_DIR}/data/datasetGeneration/data-generator.py"  # Your dataset generator script

# Where the generated datasets will go
DATA_DIR="${BASE_DIR}/data/datasets"
mkdir -p "$DATA_DIR"

# === Dataset combinations ===
# Each line has: n_examples n_features n_clusters
COMBOS=(
  "1000 2 3"
  "5000 5 4"
)

# === Main loop ===
i=1
for combo in "${COMBOS[@]}"; do
  read -r N_EXAMPLES N_FEATURES N_CLUSTERS <<< "$combo"

  # Create dataset directory (e.g., d_1, d_2, ...)
  DATASET_DIR="${DATA_DIR}/d_${i}"
  mkdir -p "$DATASET_DIR"

  echo "Generating dataset ${DATASET_DIR} (examples=$N_EXAMPLES, features=$N_FEATURES, clusters=$N_CLUSTERS)"

  # Run Python generator
  python3 "$PYTHON_SCRIPT" \
    -s "$N_EXAMPLES" \
    -f "$N_FEATURES" \
    -k "$N_CLUSTERS" \
    -o "${DATASET_DIR}/em_dataset.csv" \
    -m "${DATASET_DIR}/em_metadata.txt" \

  if [ $? -ne 0 ]; then
    echo "Dataset generation failed for ${DATASET_DIR}"
  else
    echo "Dataset created in ${DATASET_DIR}"
  fi

  i=$((i + 1))
done

echo "All dataset generations completed."