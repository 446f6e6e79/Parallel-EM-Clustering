#define _GNU_SOURCE


#include "headers/main.h"

#define MAX_ITER 100

double gaussian(double x, double mean, double var) {
    // Guard to avoid division by zero
    if (var <= 0.0) {
        var = 1e-12;
    }
    double coeff = 1.0 / sqrt(2.0 * M_PI * var);
    double expo  = exp(-( (x - mean)*(x - mean) ) / (2.0 * var));
    return coeff * expo;
}

double gaussian_multi(double *x, const double *mean, const double *var, int D) {
    double log_prob = 0.0;
    for(int i = 0; i < D; i++) {
        double v = var[i];
        // Guard to avoid division by zero
        if(v <= 0.0) v = 1e-12; 
        double diff = x[i] - mean[i];
        // log gaussian componente i
        log_prob += -0.5 * (log(2.0 * M_PI * v) + (diff*diff) / v);
    }
    return exp(log_prob);
}

/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file> [output_labels_file]
*/
int main(int argc, char **argv) {
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    double *X = NULL;                   // X[N * D] Vector of data points
    double *mu = NULL;                  // mu[k] Vector of cluster means
    double *sigma = NULL;               // sigma[k] Vector of cluster variances
    double *pi = NULL;                  // pi[k] Vector of mixture weights
    double *gamma = NULL;               // gamma[N * K] Responsibilities vector, where gamma[i*K + k] is the responsibility of cluster k for data point i
    
    int *predicted_labels = NULL;       // Predicted cluster labels
    int *ground_truth_labels = NULL;    // Ground truth labels
    
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
    printf("Metadata: samples N=%d, features D=%d, clusters K=%d\n", N, D, K);

    // Allocate buffers
    X = malloc(N * D * sizeof(double));
    predicted_labels = malloc(N * sizeof(int));
    ground_truth_labels = malloc(N * sizeof(int));
    mu = malloc(K * sizeof(double));
    sigma = malloc(K * sizeof(double));
    pi = malloc(K * sizeof(double));
    gamma = malloc(N * K * sizeof(double));

    // Check that all allocations were successful
    if(!X || !predicted_labels || !ground_truth_labels || !mu || !sigma || !pi || !gamma){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma);
        return 1;
    }   

    // Read dataset
    if(read_dataset(filename, D, N, X, ground_truth_labels) != 0){
        fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma);
        return 1;
    }
    printf("dataset read -> %d rows\n", N);

    // test print first few rows
    for(int i = 0; i < 5 && i < N; i++){
        printf("Row %d: ", i);
        for(int f = 0; f < D; f++){
            printf("%lf ", X[i * D + f]);
        }
        printf("Label=%d\n", ground_truth_labels[i]);
    }

    
    // Initialization of the parameters
    for (int k = 0; k < K; k++) {
        // Initialize means to random data points
        mu[k] = X[(rand() % N)];
        // Initialize variances to 1.0
        sigma[k] = 1.0;
        // Initialize mixture weights uniformly (each cluster has equal weight 1 / K)
        pi[k] = 1.0 / K;
    }

    /*
        EM loop:
            - E-step: compute responsibilities (gamma)
            - M-step: update parameters (mu, sigma, pi)
        The loop runs until MAX_ITER is reached
    */
    for (int iter = 0; iter < MAX_ITER; iter++) {
        
        // E-step
        for (int i = 0; i < N; i++) {
            double denom = 0.0;
            for (int k = 0; k < K; k++) {
                gamma[i*K + k] = pi[k] * gaussian(X[i], mu[k], sigma[k]);
                denom += gamma[i*K + k];
            }
            // Guard to avoid division by zero
            if (denom == 0.0 || isnan(denom)) denom = 1e-12;
            for (int k = 0; k < K; k++) gamma[i*K + k] /= denom;
        }
        
        // M-step
        for (int k = 0; k < K; k++) {
            double Nk = 0.0;                // Sum of responsibilities for cluster k
            double mu_num = 0.0;            // Weighted sum of data points (for mean calculation)
            double var_num = 0.0;           // Weighted sum of squared deviations (for variance calculation) 
            // Sum over all data points for Nk and mu_num
            for (int i = 0; i < N; i++) {
                Nk += gamma[i*K + k];
                mu_num += gamma[i*K + k] * X[i];
            }
            // Guard to avoid division by zero
            if (Nk <= 0.0 || isnan(Nk)) Nk = 1e-12;
            // Update mean
            mu[k] = mu_num / Nk;
            // Compute variance
            for (int i = 0; i < N; i++) {
                var_num += gamma[i*K + k] * (X[i] - mu[k]) * (X[i] - mu[k]);
            }
            // Update variance and mixture weight
            sigma[k] = var_num / Nk;
            pi[k] = Nk / N;
        }
    }
    
    // --- Clustering assignment ---
    for (int i = 0; i < N; i++) {
        // Find the cluster with the highest responsibility
        int best_k = 0;
        double best_val = gamma[i*K + 0];
        for (int k = 1; k < K; k++) {
            // Update best_k if current responsibility is higher
            if (gamma[i*K + k] > best_val) {
                best_val = gamma[i*K + k];
                best_k = k;
            }
        }
        predicted_labels[i] = best_k;
    }

    // Print final parameters 
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

    // Free al the allocated memory
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma);
    return 0;
}
