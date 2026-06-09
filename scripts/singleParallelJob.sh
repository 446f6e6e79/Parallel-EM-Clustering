#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb 
#PBS -l place=pack:excl
#PBS -l walltime=06:00:00
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 1 parallel-em-clustering/bin/EM_Clustering \
    -i parallel-em-clustering/data/raw/em_dataset.csv \
    -m parallel-em-clustering/data/raw/em_metadata.txt \
    -o parallel-em-clustering/data/algorithm_results/em_validation.csv

