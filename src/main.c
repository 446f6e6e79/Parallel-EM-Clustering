#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "headers/main.h"
#include "headers/file_io.h"

#define MAX_ITER 100

double gaussian(double x, double mean, double var) {
    if (var <= 0.0) {
        var = 1e-12;
    }
    double coeff = 1.0 / sqrt(2.0 * M_PI * var);
    double expo  = exp(-( (x - mean)*(x - mean) ) / (2.0 * var));
    return coeff * expo;
}

/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file> [output_labels_file]
*/
int main(int argc, char **argv) {
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters   
    double *X = NULL;                   // Data matrix
    int *predicted_labels = NULL;       // Predicted cluster labels
    int *ground_truth_labels = NULL;    // Ground truth labels
    double *mu = NULL;                  // Cluster means
    double *sigma = NULL;               // Cluster variances
    double *pi = NULL;                  // Mixture weights
    double *resp = NULL;                // Responsibilities
    
    // Check command line arguments
    if(argc < 3 || argc > 4){
        fprintf(stderr, "Usage: %s <dataset_file> <metadata_file> [output_labels_file]\n", argv[0]);
        return 1;
    }
    if(argv[1] == NULL || argv[2] == NULL || (argc > 3 && argv[3] == NULL)){
        fprintf(stderr, "Dataset file and metadata file must be provided\n");
        return 1;
    }

    // Get filenames from arguments
    const char *filename = argv[1];
    const char *metadata_filename = argv[2];
    const char *output_labels_file = (argc > 3) ? argv[3] : NULL;

    // Read metadata from metadata file
    int meta_status = read_metadata(metadata_filename, &N, &D, &K);
    if(meta_status != 0){
        fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
        return 1;
    }
    printf("Metadata: samples=%d, features=%d, clusters=%d\n", N, D, K);

    // Allocate buffers
    X = malloc(N * D * sizeof(double));
    predicted_labels = malloc(N * sizeof(int));
    ground_truth_labels = malloc(N * sizeof(int));
    mu = malloc(K * sizeof(double));
    sigma = malloc(K * sizeof(double));
    pi = malloc(K * sizeof(double));
    resp = malloc((size_t)N * K * sizeof(double));

    // Check that all allocations were successful
    if(!X || !predicted_labels || !ground_truth_labels || !mu || !sigma || !pi || !resp){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&resp);
        return 1;
    }   

    // Read dataset
    if(read_dataset(filename, D, N, X, ground_truth_labels) != 0){
        fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&resp);
        return 1;
    }
    printf("dataset read -> %d rows\n", N);

    /* TODO:check from here on */
    // test print first few rows
    for(int i=0; i<5 && i<N; i++){
        printf("Row %d: ", i);
        for(int f=0; f<D; f++){
            printf("%lf ", X[i * D + f]);
        }
        printf("Label=%d\n", ground_truth_labels[i]);
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
    }

    // --- Clustering assignment ---
    for (int i = 0; i < N; i++) {
        int best_k = 0;
        double best_val = resp[i*K + 0];
        for (int k = 1; k < K; k++) {
            if (resp[i*K + k] > best_val) {
                best_val = resp[i*K + k];
                best_k = k;
            }
        }
        predicted_labels[i] = best_k;
    }

    // --- Print final parameters ---
    for (int k = 0; k < K; k++) {
        printf("Cluster %d: mu=%.3f sigma=%.3f pi=%.3f\n", k, mu[k], sqrt(sigma[k]), pi[k]);
    }

    // Write final cluster assignments to file to validate
    if (output_labels_file){
        printf("Writing labels to file: %s\n", output_labels_file);
        int write_status = write_labels_info(output_labels_file, predicted_labels, ground_truth_labels, N);
        if(write_status != 0){
            fprintf(stderr, "Failed to write labels to file: %s\n", output_labels_file);
        }
    }

    // free memory
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&resp);
    return 0;
}
