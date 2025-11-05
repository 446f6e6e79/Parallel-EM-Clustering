#!/bin/bash
# Script to delete temporary *.sh.e* and *.sh.o* files recursively
# Those files are created by PBS and represent respetively the error and output logs of jobs

# Start in the directory where the script is launched
BASE_DIR=$(pwd)

echo "Searching for temporary files in: $BASE_DIR"

# Find and delete matching files
find "$BASE_DIR" -type f \( -name "*.sh.e*" -o -name "*.sh.o*" \) -print -delete

echo "Cleanup completed."
