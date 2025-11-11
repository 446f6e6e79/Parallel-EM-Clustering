#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "utils.h"

void scatter_dataset(double *X, double *local_X, int N, int local_N, int D, int rank, int size);
void gather_dataset(int *local_predicted_labels, int *predicted_labels, int N, int local_N, int rank, int size);
void broadcast_clusters_parameters(double *mu, double *sigma, double *pi, int K, int D);
void compute_counts_displs(int N, int size, int factor, int *counts, int *displs);

#endif

