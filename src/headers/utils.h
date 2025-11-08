#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Cleanup helper */
void reset_accumulators(double *N_k, double *mu_k, double *sigma_k, int K, int D);
void safe_cleanup(double **X, int **predicted_labels, int **ground_truth_labels, double **mu, double **sigma, double **pi, double **resp, double **N_k, double **mu_k, double **sigma_k);

#endif
