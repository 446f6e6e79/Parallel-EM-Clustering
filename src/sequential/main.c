#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../headers/file_io.h"

#define MAX_ITER 100

double gaussian(double x, double mean, double var) {
    return (1.0 / sqrt(2 * M_PI * var)) * exp(- (x - mean)*(x - mean) / (2 * var));
}
/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file>
*/
int main(int argc, char **argv) {
    // Check command line arguments
    if(argc < 3){
        fprintf(stderr, "Usage: %s <dataset_file> <metadata_file>\n", argv[0]);
        return 1;
    }
    if(argv[1] == NULL || argv[2] == NULL){
        fprintf(stderr, "Dataset file and metadata file must be provided\n");
        return 1;
    }

    // Get filenames from arguments
    const char *filename = argv[1];
    const char *metadata_filename = argv[2];

    // Read metadata from metadata file
    // N = samples, D = features, K = clusters
    int N = 0, D = 0, K = 0;
    int meta_status = read_metadata(metadata_filename, &N, &D, &K);
    if(meta_status != 1){
        fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
        return 1;
    }
    printf("Metadata: samples=%d, features=%d, clusters=%d\n", N, D, K);

    // Allocate buffers
    double *X = malloc(N * D * sizeof(double));
    int *labels_buffer = malloc(N * sizeof(int));
    double *mu = malloc(K * sizeof(double));
    double *sigma = malloc(K * sizeof(double));
    double *pi = malloc(K * sizeof(double));
    double *resp = malloc((size_t)N * K * sizeof(double));

    // Read dataset
    int n_read = read_dataset(filename, D, N, X, labels_buffer);
    if(n_read != 1){
        fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
        free(X);
        free(labels_buffer);
        return 1;
    }
    printf("dataset read -> %d rows\n", N);

    // test print first few rows
    for(int i=0; i<5 && i<N; i++){
        printf("Row %d: ", i);
        for(int f=0; f<D; f++){
            printf("%lf ", X[i * D + f]);
        }
        printf("Label=%d\n", labels_buffer[i]);
    }


    // --- Initialize parameters ---
    for (int k = 0; k < K; k++) {
        mu[k] = X[(rand() % N) * D];
        sigma[k] = 1.0;
        pi[k] = 1.0 / K;
    }

    // --- EM loop ---
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // E-step
        for (int i = 0; i < N; i++) {
            double denom = 0.0;
            for (int k = 0; k < K; k++) {
                resp[i*K + k] = pi[k] * gaussian(X[i * D], mu[k], sigma[k]);
                denom += resp[i*K + k];
            }
            for (int k = 0; k < K; k++) resp[i*K + k] /= denom;
        }

        // M-step
        for (int k = 0; k < K; k++) {
            double Nk = 0.0, mu_num = 0.0, var_num = 0.0;
            for (int i = 0; i < N; i++) {
                Nk += resp[i*K + k];
                mu_num += resp[i*K + k] * X[i * D];
            }
            mu[k] = mu_num / Nk;
            for (int i = 0; i < N; i++) {
                var_num += resp[i*K + k] * (X[i * D] - mu[k]) * (X[i * D] - mu[k]);
            }
            sigma[k] = var_num / Nk;
            pi[k] = Nk / N;
        }

        // --- Clustering assignment ---
        int *cluster_buffer = malloc(N * sizeof(int));
        int *cluster_count = malloc(K * sizeof(int));

        for (int i = 0; i < N; i++) {
            int best_k = 0;
            double best_val = resp[i*K + 0];
            for (int k = 1; k < K; k++) {
                if (resp[i*K + k] > best_val) {
                    best_val = resp[i*K + k];
                    best_k = k;
                }
            }
            cluster_buffer[i] = best_k;
            cluster_count[best_k]++;
        }

        // Summary
        for (int k = 0; k < cluster_buffer; k++)
            printf("Cluster %d points = %d\n", k, cluster_count[k]);

        free(cluster_buffer);
        free(cluster_count);
    }

    // --- Print final parameters ---
    for (int k = 0; k < K; k++) {
        printf("Cluster %d: mu=%.3f sigma=%.3f pi=%.3f\n", k, mu[k], sqrt(sigma[k]), pi[k]);
    }

    // free memory
    free(X);
    free(labels_buffer);
    free(mu);
    free(sigma);
    free(pi);
    free(resp); 
    return 0;
}
