#!/bin/bash
#PBS -l select=1:ncpus=2:mem=16gb 
#PBS -l place=pack:excl
#PBS -l walltime=06:00:00
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 2 Parallel-EM-Clustering/bin/EM_clustering \
    -i Parallel-EM-Clustering/data/em_dataset.csv \
    -m Parallel-EM-Clustering/data/em_metadata.txt \
    -o Parallel-EM-Clustering/data/em_validation.csv
