#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "types.h"

/* Cleanup helper */
void reset_accumulators(Accumulators *acc, Metadata *metadata);
void parallel_reset_accumulators(Accumulators *acc, Accumulators *local_acc, Metadata *metadata);
int alloc_cluster_params(ClusterParams *params, Metadata *metadata);
void free_cluster_params(ClusterParams *params);
int alloc_accumulators(Accumulators *acc, Metadata *metadata);
void free_accumulators(Accumulators *acc);
void safe_cleanup(ClusterParams *cluster_params, Accumulators *cluster_acc, double **X, int **predicted_labels, int **ground_truth_labels, double **resp);
void start_timer(double *t);
void stop_timer(double *t, double *accumulator);
void initialize_timers(Timers_t *timers);
int parseParameter(int argc, char **argv, InputParams_t *inputParams);
#endif
