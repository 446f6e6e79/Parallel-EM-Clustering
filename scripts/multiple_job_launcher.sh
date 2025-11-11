#!/bin/bash

# === Common parameters for multiple job submission ===
BASE_DIR="$HOME/Parallel-EM-Clustering"    # Base directory for the project
WALLTIME="06:00:00"
QUEUE="short_cpuQ"
MEM="16gb"
PLACEMENT="pack:excl"
EXECUTABLE="${BASE_DIR}/bin/EM_Clustering"  # Path to executable
TEMPLATE="${BASE_DIR}/scripts/job_template.sh"
DATASETS_DIR="${BASE_DIR}/data/datasets"

# Create an output dir to place the generated jobs
OUTPUT_DIR="${BASE_DIR}/jobs"
mkdir -p "$OUTPUT_DIR"

# === Automatically detect dataset directories ===
# We assume each dataset is in a directory named d_1, d_2, ...
DATASETS=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "d_*" | sort))

if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No dataset directories found in $DATASETS_DIR (expected names like d_1, d_2, ...)"
  exit 1
fi

# === Node/CPU combinations ===
COMBOS=(
  "1:1"
  "1:2"
  "1:4"
  "1:8"
  "2:8"
  "2:16"
  "4:16"
)

# === Repeat all submissions 5 times ===
for run in {1..5}; do
  echo "=== Starting iteration $run ==="

  for DATA_DIR in "${DATASETS[@]}"; do
    dataset_name=$(basename "$DATA_DIR")
    PARAMETERS="${DATA_DIR}/em_dataset.csv ${DATA_DIR}/em_metadata.txt $BASE_DIR/data/execution_info.csv"

    echo "Processing dataset: ${dataset_name} (Run $run)"

    # Check that required files exist
    for f in ${PARAMETERS}; do
      if [ ! -f "$f" ]; then
        echo "Missing file: $f — skipping $dataset_name"
        continue 2
      fi
    done

    for combo in "${COMBOS[@]}"; do
      IFS=":" read -r NODES NCPUS <<< "$combo"
      NP=$(( NODES * NCPUS ))

      JOB_SCRIPT="${OUTPUT_DIR}/job_${dataset_name}run${run}${NODES}n_${NCPUS}c.sh"

      sed "s|__EXECUTABLE__|$EXECUTABLE|g; \
           s|__PLACEMENT__|$PLACEMENT|g; \
           s|__NODES__|$NODES|g; \
           s|__NCPUS__|$NCPUS|g; \
           s|__MEM__|$MEM|g; \
           s|__WALLTIME__|$WALLTIME|g; \
           s|__QUEUE__|$QUEUE|g; \
           s|__NP__|$NP|g; \
           s|__PARAMETERS__|$PARAMETERS|g" \
           "$TEMPLATE" > "$JOB_SCRIPT"

      chmod +x "$JOB_SCRIPT"

      echo "Generated $JOB_SCRIPT (Run=$run, Dataset=$dataset_name, NODES=$NODES, NCPUS=$NCPUS, NP=$NP)"

      qsub "$JOB_SCRIPT"

      rm "$JOB_SCRIPT"
    done
  done
done