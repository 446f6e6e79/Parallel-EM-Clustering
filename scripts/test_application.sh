#!/bin/bash
set -euo pipefail

# === Common parameters ===
BASE_DIR="$(pwd)"
PARALLEL_EXECUTABLE="${BASE_DIR}/bin/EM_Clustering_debug"
SEQUENTIAL_EXECUTABLE="${BASE_DIR}/bin/EM_Sequential_debug"
DATASETS_DIR="${BASE_DIR}/test"

OUTPUT_DIR="${DATASETS_DIR}/local_runs"
LOG_DIR="${OUTPUT_DIR}/logs"
NP=4  # Number of parallel processes

ensure_exec() {
  # Parameters:
  local target="$1"
  local make_target="$2"
  # If the executable does not exist, build it
  if [[ ! -x "$target" ]]; then
    echo "Missing debug executable: $target (building: make $make_target)..."
    (cd "$BASE_DIR" && make "$make_target") || {
      echo "Failed to build $make_target" >&2
      exit 1
    }
    # Verify it was built
    if [[ ! -x "$target" ]]; then
      echo "Executable still missing after build: $target" >&2
      exit 1
    fi
  fi
}

echo "Checking debug executables..."
# Ensure the parallel debug executable is built
ensure_exec "$PARALLEL_EXECUTABLE" debug
# Ensure the sequential debug executable is built
ensure_exec "$SEQUENTIAL_EXECUTABLE" sequential-debug
echo "Executables OK."

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

DATASETS=($(find "$DATASETS_DIR" -maxdepth 1 -type d -name "t_*" | sort))
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No test dataset directories found in $DATASETS_DIR"
  exit 1
fi

for DATA_DIR in "${DATASETS[@]}"; do
    dataset_name=$(basename "$DATA_DIR")
    INPUT_FILE="${DATA_DIR}/em_dataset.csv"
    META_FILE="${DATA_DIR}/em_metadata.txt"

    for f in "$INPUT_FILE" "$META_FILE"; do
        if [ ! -f "$f" ]; then
            echo "Missing file: $f — skipping $dataset_name"
            continue 2
        fi
    done

    PARAMETERS="-i $INPUT_FILE -m $META_FILE"

    echo "Running dataset: $dataset_name"

    SEQ_LOG="${LOG_DIR}/${dataset_name}_sequential.log"
    echo "  Sequential..."
    "$SEQUENTIAL_EXECUTABLE" $PARAMETERS > "$SEQ_LOG" 2>&1 || echo "  Sequential failed (see log)"

    PAR_LOG="${LOG_DIR}/${dataset_name}_parallel.log"
    echo "  Parallel ($NP proc)..."
    mpirun -np "$NP" "$PARALLEL_EXECUTABLE" $PARAMETERS > "$PAR_LOG" 2>&1 || echo "  Parallel failed (see log)"

    echo "  Logs: $SEQ_LOG | $PAR_LOG"
    echo
done

echo "All tests completed. Logs in: $LOG_DIR"