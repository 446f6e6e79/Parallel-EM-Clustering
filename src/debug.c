#ifdef DEBUG
#include <stdio.h>
#include <math.h>
#include "headers/debug.h"
#include "headers/io_utils.h"

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

/*
    Print intermediate results to a file for debugging purposes.
    Parameters:
        - filename: Path to the debug output file
        - X: Dataset buffer
        - predicted_labels: Predicted labels buffer
        - real_labels: Ground truth labels buffer
        - metadata: Metadata structure
        - cluster_params: Cluster parameters structure
        - iteration: Current iteration number
*/
void debug_print_intermediate_results(const char *filename, double *X, int *predicted_labels, int *real_labels, Metadata *metadata, ClusterParams *cluster_params, int iteration){
    // Check that the filename was passed as a parameter
    if(filename == NULL){
        fprintf(stderr, "Debug file path not provided.\n");
        return;
    }
    // Otherwise, call the function to write data to file
    if(write_labels_info(filename, X, predicted_labels, real_labels, metadata, cluster_params, iteration) != 0){
        fprintf(stderr, "Failed to write debug intermediate results to file: %s\n", filename);
    }
}
#endif