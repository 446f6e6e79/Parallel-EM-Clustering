#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb 
#PBS -l place=pack:excl
#PBS -l walltime=06:00:00
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 1 Parallel-EM-Clustering/bin/EM_Clustering Parallel-EM-Clustering/data/em_dataset.csv Parallel-EM-Clustering/data/execution_info.csv Parallel-EM-Clustering/data/em_metadata.txt
