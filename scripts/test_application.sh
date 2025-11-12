#!/bin/bash

# === Common parameters ===
BASE_DIR="$(pwd)"
PARALLEL_EXECUTABLE="${BASE_DIR}/bin/EM_Clustering"
SEQUENTIAL_EXECUTABLE="${BASE_DIR}/bin/EM_Sequential"
DATASETS_DIR="${BASE_DIR}/test"

# Directory for logging the output of local runs
OUTPUT_DIR="${DATASETS_DIR}/local_runs"
LOG_DIR="${OUTPUT_DIR}/logs"

NP=4  # Number of parallel processes

# Create a clean log directory
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# === Detect datasets ===
DATASETS=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "t_*" | sort))
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No test dataset directories found in $DATASETS_DIR"
  exit 1
fi

# === Main loop ===
for DATA_DIR in "${DATASETS[@]}"; do
    dataset_name=$(basename "$DATA_DIR")
    INPUT_FILE="${DATA_DIR}/em_dataset.csv"
    META_FILE="${DATA_DIR}/em_metadata.txt"

    # Check for required files
    for f in "$INPUT_FILE" "$META_FILE"; do
        if [ ! -f "$f" ]; then
            echo "Missing file: $f — skipping $dataset_name"
            continue 2
        fi
    done

    PARAMETERS="-i $INPUT_FILE -m $META_FILE"

    echo "Running dataset: $dataset_name"

    # --- Sequential run ---
    SEQ_LOG="${LOG_DIR}/${dataset_name}_sequential.log"
    echo "  Sequential run..."
    # Redirect the output to the log file
    "$SEQUENTIAL_EXECUTABLE" $PARAMETERS > "$SEQ_LOG" 2>&1
    echo "  Sequential run completed. Log: $SEQ_LOG"

    # --- Parallel run ---
    PAR_LOG="${LOG_DIR}/${dataset_name}_parallel.log"
    echo "  Parallel run with $NP processes..."
    # Redirect the output to the log file
    mpirun -np "$NP" "$PARALLEL_EXECUTABLE" $PARAMETERS > "$PAR_LOG" 2>&1
    echo "  Parallel run completed. Log: $PAR_LOG"
    echo
done

echo "All tests completed. Logs saved in: $LOG_DIR"
