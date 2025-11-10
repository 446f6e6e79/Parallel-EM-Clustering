#ifndef DEBUG_H
#define DEBUG_H
#include <stdio.h>
#include <math.h>

void debug_print_first_samples(int N, int D, double *X, int *ground_truth_labels);
void debug_print_scatter(int local_N, int D, double *local_X, int rank);
void debug_print_cluster_params(int K, int D, double *mu, double *sigma, double *pi);

#endif 