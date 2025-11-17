#ifndef EM_H
#define EM_H

#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "utils.h"

#define LOG_2PI 1.8378770664093453         // log(2*pi)
#define GUARD_VALUE 1e-12                  // minimum variance guard

void m_step( double *X, Metadata *metadata, ClusterParams *cluster_params, Accumulators *acc, double *gamma);
void m_step_parallelized(double *local_X, int local_N, Metadata *metadata, ClusterParams *cluster_params, Accumulators *cluster_acc, Accumulators *local_cluster_acc, double *local_gamma);
double e_step(double *X, int N, Metadata *metadata, ClusterParams *cluster_params, double *gamma);
void init_params(double *X, Metadata *metadata, ClusterParams *cluster_params);
double gaussian_multi_diag(double *x, double *mu, double *sigma, int D);  
void compute_clustering(double *gamma, int N, int K, int *predicted_labels);
int check_convergence(double *prev_log_likelihood, double *curr_log_likelihood, double threshold);

#endif
