#ifndef EM_H
#define EM_H

#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "utils.h"

#define LOG_2PI 1.8378770664093453         // log(2*pi)
#define GUARD_VALUE 1e-12                  // minimum variance guard

void m_step( double *X, int N, int D, int K, double *gamma, double *mu, double *sigma, double *pi, double *N_k, double *mu_k, double *sigma_k);
void m_step_parallelized(double *local_X, int N, int local_N, int D, int K, double *local_gamma, double *mu, double *sigma, double *pi, double *N_k, double *mu_k, double *sigma_k, int rank);
void e_step(double *X, int N, int D, int K, double *mu, double *sigma, double *pi, double *gamma);
void init_params(double *X, int N, int D, int K, double *mu, double *sigma, double *pi);
double gaussian_multi_diag(double *x, double *mu, double *sigma, int D);  
void compute_clustering(double *gamma, int N, int K, int *predicted_labels);

#endif
