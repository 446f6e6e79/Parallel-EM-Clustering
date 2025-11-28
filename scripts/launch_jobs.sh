#!/bin/bash
SHORT_QUEUE="short_HPC4DS"
SHORT_QUEUE_LIMIT=100

LONG_QUEUE="common_cpuQ"
LONG_QUEUE_LIMIT=10

JOB_DIR="$HOME/Parallel-EM-Clustering/jobs"
SHORT_JOB_DIR="${JOB_DIR}/short"
LONG_JOB_DIR="${JOB_DIR}/long"

# === Submit Short queue jobs ===
# Check how many short jobs are currently running
ACTIVE_SHORT_JOBS=$(qstat -u "$USER" | grep -E '^[0-9]+' | grep "short" | wc -l)
echo "Currently active short jobs: $ACTIVE_SHORT_JOBS"

# Compute how many more jobs we can submit
AVAILABLE_SHORT_SLOTS=$(( SHORT_QUEUE_LIMIT - ACTIVE_SHORT_JOBS ))
echo "Available slots in short queue: $AVAILABLE_SHORT_SLOTS"

# For each job in the short job directory, submit if we have available slots
for JOB_SCRIPT in "$SHORT_JOB_DIR"/*.sh; do
  if [ "$AVAILABLE_SHORT_SLOTS" -gt 0 ]; then
    echo "Submitting $JOB_SCRIPT (available slots: $AVAILABLE_SHORT_SLOTS)"
    qsub "$JOB_SCRIPT" && rm "$JOB_SCRIPT"
    AVAILABLE_SHORT_SLOTS=$(( AVAILABLE_SHORT_SLOTS - 1 ))
  else
    echo "No available slots in short queue. Stopping submission"
    break
  fi
done
# === Submit Long queue jobs ===
# Check how many long jobs are currently running
ACTIVE_LONG_JOBS=$(qstat -u "$USER" | grep -E '^[0-9]+' | grep "common" | wc -l)
echo "Currently active long jobs: $ACTIVE_LONG_JOBS"

# Compute how many more jobs we can submit
AVAILABLE_LONG_SLOTS=$(( LONG_QUEUE_LIMIT - ACTIVE_LONG_JOBS ))
echo "Available slots in long queue: $AVAILABLE_LONG_SLOTS"

# For each job in the long job directory, submit if we have available slots
for JOB_SCRIPT in "$LONG_JOB_DIR"/*.sh; do
  if [ "$AVAILABLE_LONG_SLOTS" -gt 0 ]; then
    echo "Submitting $JOB_SCRIPT (available slots: $AVAILABLE_LONG_SLOTS)"
    qsub "$JOB_SCRIPT" && rm "$JOB_SCRIPT"
    AVAILABLE_LONG_SLOTS=$(( AVAILABLE_LONG_SLOTS - 1 ))
  else
    echo "No available slots in long queue. Stopping submission"
    break
  fi
done
echo "Job submission process completed."
