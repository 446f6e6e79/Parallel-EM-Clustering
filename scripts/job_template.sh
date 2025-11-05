#!/bin/bash
#PBS -l select=__NODES__:ncpus=__NCPUS__:mem=__MEM__ 
#PBS -l place=__PLACEMENT__
#PBS -l walltime=__WALLTIME__
#PBS -q __QUEUE__

module load mpich-3.2
mpirun.actual -n __NP__ __EXECUTABLE__ __PARAMETERS__
