#!/bin/bash

OUTPUT_DIR="$HOME/Parallel-EM-Clustering/jobs"

# Launch jobs until 30 active jobs are reached
for JOB_SCRIPT in "$OUTPUT_DIR"/*.sh; do
  # Count active jobs
  ACTIVE_JOBS=$(qstat -u "$USER" | grep -E '^[0-9]+' | wc -l)

  if [ "$ACTIVE_JOBS" -lt 30 ]; then
    echo "Submitting $JOB_SCRIPT (active jobs: $ACTIVE_JOBS)"
    qsub "$JOB_SCRIPT" && rm "$JOB_SCRIPT"
  else
    echo "Reached 30 active jobs. Stopping launcher."
    exit 0
  fi
done

echo "All job scripts submitted (or skipped if limit reached)."
