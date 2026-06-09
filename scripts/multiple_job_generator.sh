#!/bin/bash
# === Parameters ===
ITERATION_PER_COMBO=3
DATASET_INITIAL_NAME="d" # As example, if your datasets are d_1, d_2, d_3 set it to d
MEM="64gb" # Memory per NODE
PLACEMENT="pack:excl"

# NODES:NCPUS:THREADS
COMBOS=( 
  "1:1:1"
  "1:1:2"
  "1:1:4"
  "1:1:8"
  "2:1:8"
  "2:2:8"
  "4:2:8"
)

SHORT_QUEUE="short_HPC4DS"
LONG_QUEUE="long_cpuQ"
SHORT_WALLTIME="06:00:00"
LONG_WALLTIME="10:00:00"

# === Common parameters ===
BASE_DIR="$HOME/parallel-em-clustering"
EXECUTABLE="${BASE_DIR}/bin/EM_Clustering"
TEMPLATE="${BASE_DIR}/scripts/job_template.sh"

DATASETS_DIR="${BASE_DIR}/data/datasets"
# Detect datasets
DATASETS=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "${DATASET_INITIAL_NAME}*" | sort))
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No dataset directories found in $DATASETS_DIR"
  exit 1
fi

# Output file for job info
OUTPUT_INFO="$BASE_DIR/data/algorithm_results/execution_info.csv"
if [ ! -f "$OUTPUT_INFO" ]; then
  echo "No output file found in $OUTPUT_INFO"
  exit
fi

# Create all the output directories for the jobs
OUTPUT_DIR="${BASE_DIR}/jobs"
mkdir -p "$OUTPUT_DIR/long" "$OUTPUT_DIR/short"

# === Define the combinations ===
for run in $(seq 1 "$ITERATION_PER_COMBO"); do
  echo "=== Generating jobs for iteration $run ==="

  for DATA_DIR in "${DATASETS[@]}"; do
    dataset_name=$(basename "$DATA_DIR")
    
    # Define the needed file parameters path
    input="${DATA_DIR}/em_dataset.csv"
    meta="${DATA_DIR}/em_metadata.txt"
    
    # Check if the all input file exists
    if [ ! -f "$input" ] || [ ! -f "$meta" ]; then
      echo "Missing required file(s) for $dataset_name — skipping"
      continue
    fi

    for combo in "${COMBOS[@]}"; do

      IFS=":" read -r NODES NCPUS THREADS <<< "$combo"

      TOTAL_PROCESSES=$(( NCPUS * THREADS ))
      NP=$((NCPUS * NODES))
      
      # Choose the queue and the walltime based on the n_process
      if [ "$TOTAL_PROCESSES" -le 2 ] && [[ "$dataset_name" == *_1 ]]; then
          QUEUE="$LONG_QUEUE"
          WALLTIME="$LONG_WALLTIME"
          CURRENT_OUTPUT_DIR="${OUTPUT_DIR}/long"
      else
          QUEUE="$SHORT_QUEUE"
          WALLTIME="$SHORT_WALLTIME"
          CURRENT_OUTPUT_DIR="${OUTPUT_DIR}/short"
      fi
      PARAMETERS="-i $input -m $meta -b $OUTPUT_INFO -n $THREADS"

      JOB_SCRIPT="${CURRENT_OUTPUT_DIR}/job_${dataset_name}-run_${run}-nodes_${NODES}-cpus_${NCPUS}-threads_${THREADS}.sh"

      sed "s|__EXECUTABLE__|$EXECUTABLE|g; \
           s|__PLACEMENT__|$PLACEMENT|g; \
           s|__NODES__|$NODES|g; \
           s|__NCPUS__|$NCPUS|g; \
           s|__THREADS__|$THREADS|g; \
           s|__MEM__|$MEM|g; \
           s|__WALLTIME__|$WALLTIME|g; \
           s|__QUEUE__|$QUEUE|g; \
           s|__TOTALP__|$TOTAL_PROCESSES|g; \
           s|__NP__|$NP|g; \
           s|__PARAMETERS__|$PARAMETERS|g" \
           "$TEMPLATE" > "$JOB_SCRIPT"

      chmod +x "$JOB_SCRIPT"
      echo "Generated $JOB_SCRIPT"
    done
  done
done
