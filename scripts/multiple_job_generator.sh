#!/bin/bash

# === Common parameters ===
BASE_DIR="$HOME/Parallel-EM-Clustering"
WALLTIME="06:00:00"
QUEUE="short_cpuQ"
MEM="16gb"
PLACEMENT="pack:excl"
EXECUTABLE="${BASE_DIR}/bin/EM_Clustering"
TEMPLATE="${BASE_DIR}/scripts/job_template.sh"
DATASETS_DIR="${BASE_DIR}/data/datasets"

OUTPUT_DIR="${BASE_DIR}/jobs"
mkdir -p "$OUTPUT_DIR"

# Detect datasets
DATASETS=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "d_*" | sort))
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No dataset directories found in $DATASETS_DIR"
  exit 1
fi

COMBOS=(
  "1:1"
  "1:2"
  "1:4"
  "1:8"
  "2:8"
  "2:16"
  "4:16"
)

for run in {1..3}; do
  echo "=== Generating jobs for iteration $run ==="

  for DATA_DIR in "${DATASETS[@]}"; do
    dataset_name=$(basename "$DATA_DIR")
    PARAMETERS="${DATA_DIR}/em_dataset.csv ${DATA_DIR}/em_metadata.txt $BASE_DIR/data/execution_info.csv"

    # Skip if any required file is missing
    missing=false
    for f in ${PARAMETERS}; do
      if [ ! -f "$f" ]; then
        echo "Missing file: $f — skipping $dataset_name"
        missing=true
        break
      fi
    done
    $missing && continue

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
      echo "Generated $JOB_SCRIPT"
    done
  done
done
