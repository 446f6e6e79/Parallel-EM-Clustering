#ifndef DEBUG_H
#define DEBUG_H

#include <stddef.h>
#include "types.h"

#ifdef DEBUG
// Debug function declared in debug.c
void debug_print_first_samples(Metadata *metadata, double *X, int *ground_truth_labels);
void debug_print_cluster_params(Metadata *metadata, ClusterParams *cluster_params);
void debug_print_scatter(int local_N, int D, double *local_X, int rank);
void debug_print_intermediate_results(const char *filename, double *X, int *predicted_labels, int *real_labels, Metadata *metadata, ClusterParams *cluster_params, int iteration);

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
void debug_print_intermediate_results(const char *filename, double *X, int *predicted_labels, int *real_labels, Metadata *metadata, ClusterParams *cluster_params, int iteration) {
    (void)filename; (void)X; (void)predicted_labels; (void)real_labels; (void)metadata; (void)cluster_params; (void)iteration;
}
#endif

#endif