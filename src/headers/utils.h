#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "types.h"

/* Cleanup helper */
void reset_accumulators(double *N_k, double *mu_k, double *sigma_k, int K, int D);
void parallel_reset_accumulators(double *N_k, double *mu_k, double *sigma_k, double *local_N_k, double *local_mu_k, double *local_sigma_k, int K, int D);
void safe_cleanup(double **X, int **predicted_labels, int **ground_truth_labels, double **mu, double **sigma, double **pi, double **resp, double **N_k, double **mu_k, double **sigma_k);
void safe_cleanup_local(double **local_N_k, double **local_mu_k, double **local_sigma_k);
void start_timer(double *t);
void stop_timer(double *t, double *accumulator);
void initialize_timers(Timers_t *timers);
#endif
