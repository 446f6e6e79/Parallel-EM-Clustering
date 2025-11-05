#!/bin/bash

# Setup common parameters for the multiple job submission
BASE_DIR="$HOME/Parallel-EM-Clustering"  # Base directory for the project
WALLTIME="06:00:00"
QUEUE="short_cpuQ"
MEM="16gb"
PLACEMENT="pack:excl"
PARAMETERS=""
EXECUTABLE="${BASE_DIR}/bin/EM_Clustering"      # Path to the executable file
TEMPLATE="${BASE_DIR}/scripts/job_template.sh"

# Output directory for generated job scripts
OUTPUT_DIR="${BASE_DIR}/jobs"

# List of "NODES:NCPUS" combinations to produce
COMBOS=(
  "1:1"    # 1 cpu, 1 node.       -> P = 1
  "1:2"    # 2 cpu, 1 node        -> P = 2
  "1:4"    # 4 cpu, 1 node        -> P = 4
  "1:8"    # 8 cpu, 1 nodes       -> P = 8
  "2:8"    # 8 cpu, 2 nodes       -> P = 16
  "2:16"   # 16 cpu, 2 nodes       -> P = 32
  "4:16"   # 16 cpu, 4 nodes       -> P = 64
)

# Loop over each combination and generate the corresponding job script
for combo in "${COMBOS[@]}"; do
  # Split the combo into NODES and NCPUS
  IFS=":" read -r NODES NCPUS <<< "$combo"
  # Compute the number of processes as NODES * NCPUS
  NP=$(( NODES * NCPUS ))

  # Generate job name dinamically, as job_run_<NODES>n_<NCPUS>c.sh
  out="job_run_${NODES}n_${NCPUS}c.sh"

  sed "s|__EXECUTABLE__|$EXECUTABLE|g; \
      s|__PLACEMENT__|$PLACEMENT|g; \
      s|__NODES__|$NODES|g; \
      s|__NCPUS__|$NCPUS|g; \
      s|__MEM__|$MEM|g; \
      s|__WALLTIME__|$WALLTIME|g; \
      s|__QUEUE__|$QUEUE|g; \
      s|__NP__|$NP|g; \
      s|__PARAMETERS__|$PARAMETERS|g" "$TEMPLATE" > "$out"


  # Make the generated script executable
  chmod +x "$out"
  echo "Generated $out (NODES=$NODES NCPUS=$NCPUS NP=$NP)"

  # Submit to PBS (uncomment to actually submit)
  qsub "$out"

  # Optionally remove the temporary file after submission
  rm "$out"
done