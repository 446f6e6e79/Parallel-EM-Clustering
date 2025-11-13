#ifdef DEBUG
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include "headers/debug.h"
#include "headers/io_utils.h"

/*
    Helper function for debug printing with newline 
*/
void debug_println(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    fputc('\n', stdout);
    fflush(stdout); // useful with MPI
    va_end(ap);
}

/*
    Print the first few samples of the dataset for debugging the read process.
    Parameters:
        - N: Number of samples
        - D: Number of features
        - X: Dataset buffer
        - ground_truth_labels: Ground truth labels buffer
*/
void debug_print_first_samples(Metadata *metadata, double *X, int *ground_truth_labels) {
    printf("\ndataset read -> %d rows\n", metadata->N);
    // test print first few rows
    for(int i = 0; i < 5 && i < metadata->N; i++){
        printf("Sample-%d: ", i);
        for(int f = 0; f < metadata->D; f++){
            printf("%lf ", X[i * metadata->D + f]);
        }
        printf("Label=%d\n", ground_truth_labels[i]);
    }
}

/*
  Print the parameters of each cluster
   Parameters:
   -metadata:
        -K: Number of clusters
        -D: Number of features
   -cluster_params: 
        -mu: (K x D) Matrix of cluster means
        -sigma: (K x D) Matrix of cluster variances
        -pi: (K) Vector of mixture weights
 */
void debug_print_cluster_params(Metadata *metadata, ClusterParams *cluster_params) {
    for (int k = 0; k < metadata->K; k++) {
        printf("Cluster %d: \n", k);
        printf("\tpi=%.6f\n", cluster_params->pi[k]);
        printf("\tmu: ");
        for (int d = 0; d < metadata->D; d++) printf("%.3f ", cluster_params->mu[k * metadata->D + d]);
        printf("\n");
        printf("\tsigma (std per-dim): ");
        for (int d = 0; d < metadata->D; d++) printf("%.3f ", sqrt(cluster_params->sigma[k * metadata->D + d]));
        printf("\n");
    }
}

/*
    Print the first few samples of every local process for debugging the data distribution.
    Parameters:
        - local_N: Number of samples
        - D: Number of features
        - local_X: Dataset buffer
        - rank: Rank of the current process
*/
void debug_print_scatter(int local_N, int D, double *local_X, int rank) {
    printf("\nProcess %d: dataset read -> %d rows\n", rank, local_N);
    // test print first few rows
    for(int i = 0; i < 5 && i < local_N; i++){
        printf("Process%d-Sample-%d: ", rank, i);
        for(int f = 0; f < D; f++){
            printf("%lf ", local_X[i * D + f]);
        }
        printf("\n");
    }
}
#endif