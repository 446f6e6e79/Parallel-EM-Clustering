#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* Cleanup helper */
void safe_cleanup(double **X, int **predicted_labels, int **ground_truth_labels, double **mu, double **sigma, double **pi, double **resp);
#endif
