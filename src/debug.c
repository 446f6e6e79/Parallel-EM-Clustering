#include "headers/debug.h"

/*
    Print the first few samples of the dataset for debugging the read process.
    Parameters:
        - N: Number of samples
        - D: Number of features
        - X: Dataset buffer
        - ground_truth_labels: Ground truth labels buffer

*/
void debug_print_first_samples(int N, int D, double *X, int *ground_truth_labels) {
    printf("dataset read -> %d rows\n\n", N);
    // test print first few rows
    for(int i = 0; i < 5 && i < N; i++){
        printf("Sample-%d: ", i);
        for(int f = 0; f < D; f++){
            printf("%lf ", X[i * D + f]);
        }
        printf("Label=%d\n", ground_truth_labels[i]);
    }
}

/*
  Print the parameters of each cluster
   Parameters:
   -K: Number of clusters
   -D: Number of features
   -mu: (K x D) Matrix of cluster means
   -sigma: (K x D) Matrix of cluster variances
   -pi: (K) Vector of mixture weights
 */
void debug_print_cluster_params(int K, int D, double *mu, double *sigma, double *pi) {
    for (int k = 0; k < K; k++) {
        printf("Cluster %d: \n", k);
        printf("\tpi=%.6f\n", pi[k]);
        printf("\tmu: ");
        for (int d = 0; d < D; d++) printf("%.3f ", mu[k * D + d]);
        printf("\n");
        printf("\tsigma (std per-dim): ");
        for (int d = 0; d < D; d++) printf("%.3f ", sqrt(sigma[k * D + d]));
        printf("\n");
    }
}