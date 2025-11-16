#!/bin/bash

# === Configuration ===
# Automatically find the base directory (where this script lives)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${BASE_DIR}/tools/dataGeneration/data-generator.py"  # Your dataset generator script

# Where the generated datasets will go
DATA_DIR="${BASE_DIR}/data/datasets"
mkdir -p "$DATA_DIR"

# Where the testing datasets will go
TESTING_DIR="${BASE_DIR}/test"
mkdir -p "$TESTING_DIR"

# === Argument parsing ===
MODE="benchmark"  # default mode

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--mode benchmark|test]"
      exit 1
      ;;
  esac
done

# === Dataset configurations ===
# Each line: n_examples n_features n_clusters
COMBOS=(
  "10000000 50 15"
  "5000000 50 15"
  "2500000 50 15"
  "1250000 50 15"
  "625000 50 15"
  "312500 50 15"
  "156250 50 15"
)

# Each line: n_examples n_features n_clusters mean1 mean2 mean3
TESTING_COMBOS=(
  "2000 1 3 30 -30 0"
  "2000 1 2 30 -30"
  "10000 2 3 30,0 30,-30 0,30"
)

# === Main logic ===
if [[ "$MODE" == "benchmark" ]]; then
  echo "Running in BENCHMARK mode..."
  i=1
  for combo in "${COMBOS[@]}"; do
    read -r N_EXAMPLES N_FEATURES N_CLUSTERS <<< "$combo"

    DATASET_SUBDIR="${DATA_DIR}/d_${i}"
    mkdir -p "$DATASET_SUBDIR"

    echo "Generating dataset ${DATASET_SUBDIR} (examples=$N_EXAMPLES, features=$N_FEATURES, clusters=$N_CLUSTERS)"

    python3 "$PYTHON_SCRIPT" \
      -s "$N_EXAMPLES" \
      -f "$N_FEATURES" \
      -k "$N_CLUSTERS" \
      -o "${DATASET_SUBDIR}/em_dataset.csv" \
      -m "${DATASET_SUBDIR}/em_metadata.txt"

    if [ $? -ne 0 ]; then
      echo "Dataset generation failed for ${DATASET_SUBDIR}"
    else
      echo "Dataset created in ${DATASET_SUBDIR}"
    fi

    i=$((i + 1))
  done

elif [[ "$MODE" == "test" ]]; then
  echo "Running in TEST mode..."
  i=1
  for combo in "${TESTING_COMBOS[@]}"; do
    # Split the line into an array
    read -r -a FIELDS <<< "$combo"

    # Extract the first three as fixed parameters
    N_EXAMPLES=${FIELDS[0]}
    N_FEATURES=${FIELDS[1]}
    N_CLUSTERS=${FIELDS[2]}

    # The remaining fields (if any) are means
    MEANS=("${FIELDS[@]:3}")

    TEST_SUBDIR="${TESTING_DIR}/t_${i}"
    mkdir -p "$TEST_SUBDIR"

    echo "Generating TEST dataset ${TEST_SUBDIR} (examples=$N_EXAMPLES, features=$N_FEATURES, clusters=$N_CLUSTERS, means=${MEANS[*]})"

    # Run Python script with dynamic means
    python3 "$PYTHON_SCRIPT" \
      -s "$N_EXAMPLES" \
      -f "$N_FEATURES" \
      -k "$N_CLUSTERS" \
      -o "${TEST_SUBDIR}/em_dataset.csv" \
      -m "${TEST_SUBDIR}/em_metadata.txt" \
      --means "${MEANS[@]}"

    if [ $? -ne 0 ]; then
      echo "Test dataset generation failed for ${TEST_SUBDIR}"
    else
      echo "Test dataset created in ${TEST_SUBDIR}"
    fi

    i=$((i + 1))
  done


else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [--mode benchmark|test]"
  exit 1
fi

echo "All dataset generations completed."
