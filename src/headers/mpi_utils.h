#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "types.h"

void scatter_dataset(double *X, double *local_X, int local_N, Metadata *metadata, int rank, int size);
void gather_dataset(int *local_predicted_labels, int *predicted_labels, int N, int local_N, int rank, int size);
void broadcast_metadata(Metadata *metadata, int rank);
void broadcast_clusters_parameters(double *mu, double *sigma, double *pi, Metadata *metadata);
void compute_counts_displs(int N, int size, int factor, int *counts, int *displs);
int compute_local_N(int N, int size, int rank);

#endif

