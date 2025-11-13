#ifndef DEBUG_H
#define DEBUG_H

#include <stddef.h>
#include "types.h"

#ifdef DEBUG
// Debug function declared in debug.c
void debug_print_first_samples(Metadata *metadata, double *X, int *ground_truth_labels);
void debug_print_cluster_params(Metadata *metadata, ClusterParams *cluster_params);
void debug_print_scatter(int local_N, int D, double *local_X, int rank);

#else
// If DEBUG is not defined, provide NOP implementations
void debug_print_first_samples(Metadata *metadata, double *X, int *ground_truth_labels){
    (void)metadata; (void)X; (void)ground_truth_labels;
}
void debug_print_cluster_params(Metadata *metadata, ClusterParams *cluster_params) {
    (void)metadata; (void)cluster_params;
}
void debug_print_scatter(int local_N, int D, double *local_X, int rank) {
    (void)local_N; (void)D; (void)local_X; (void)rank;
}
#endif

#endif