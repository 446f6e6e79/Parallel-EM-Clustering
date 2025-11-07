#define _GNU_SOURCE

#include "headers/main.h"

#define MAX_ITER 100

/*
    Multivariate Gaussian probability density function with diagonal covariance matrix.
    p(x | mu, Sigma_diag) =
            exp( -0.5 * ( D*log(2*pi) 
                         + sum_d log(sigma[d])
                         + sum_d (x[d]-mu[d])^2 / sigma[d] ) )
*/
double gaussian_multi_diag(double *x, double *mu, double *sigma, int D) {
    double logdet = 0.0;
    double quad = 0.0;
    for (int d = 0; d < D; d++) {
        double var_d = sigma[d];
        // Guard to avoid division by zero
        if (var_d <= 0.0) var_d = EPS_VAR;
        logdet += log(var_d);
        double diff = x[d] - mu[d];
        quad += (diff * diff) / var_d;
    }
    double exponent = -0.5 * ( D * LOG_2PI + logdet + quad );
    return exp(exponent);
}

/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file> [output_labels_file]
*/
int main(int argc, char **argv) {
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    int max_line_size;                  // Maximum number of character in a line in the dataset file
    
    double *X = NULL;                   // X[N * D] Vector of data points
    double *mu = NULL;                  // mu[k] Vector of cluster means
    double *sigma = NULL;               // sigma[k] Vector of cluster variances
    double *pi = NULL;                  // pi[k] Vector of mixture weights
    double *gamma = NULL;               // gamma[N * K] Responsibilities vector, where gamma[i*K + k] is the responsibility of cluster k for data point i
    
    int *predicted_labels = NULL;       // Predicted cluster labels
    int *ground_truth_labels = NULL;    // Ground truth labels

    double *N_k;                        // Sum of responsibilities per cluster
    double *mu_k;                       // Weighted sums for means
    double *sigma_k;                    // Weighted sums for variances
    
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
    int meta_status = read_metadata(metadata_filename, &N, &D, &K, &max_line_size);
    if(meta_status != 0){
        fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
        return 1;
    }
    printf("Metadata: samples N=%d, features D=%d, clusters K=%d\n", N, D, K);

    // Allocate buffers
    X = malloc(N * D * sizeof(double));
    predicted_labels = malloc(N * sizeof(int));
    ground_truth_labels = malloc(N * sizeof(int));
    mu = malloc(K * D * sizeof(double)); // changed
    sigma = malloc(K * D * sizeof(double)); // changed
    pi = malloc(K * sizeof(double));
    gamma = malloc(N * K * sizeof(double));
    N_k = malloc((size_t)K * sizeof(double));              
    mu_k = malloc((size_t)K * D * sizeof(double)); // new  
    sigma_k = malloc((size_t)K * D * sizeof(double));   // new   

    // Check that all allocations were successful
    if(!X || !predicted_labels || !ground_truth_labels || !mu || !sigma || !pi || !gamma || !N_k || !mu_k || !sigma_k){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma,&N_k,&mu_k,&sigma_k);
        return 1;
    }   

    // Read dataset
    if(read_dataset(filename, D, N, max_line_size, X, ground_truth_labels) != 0){
        fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma,&N_k,&mu_k,&sigma_k);
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

    // Seed RNG
    srand(0);
    
    // Initialization of the parameters
    for (int k = 0; k < K; k++) { 
        // Choose a random index between 0 and N-1
        int r = rand() % N;
        for (int d=0; d<D; d++) {
            // Initialize means to random data points
            mu[k*D + d] = X[r*D + d];
            // Initialize variances to 1.0
            sigma[k*D + d] = 1.0;
        }
        // Initialize mixture weights uniformly (each cluster has equal weight 1 / K)
        pi[k] = 1.0 / (double) K;
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
            double *x = &X[i*D]; // pointer to the i-th data point
            for (int k = 0; k < K; k++) {
                double *mu_k = &mu[k*D];         // Vector mean of cluster k
                double *sigma_k = &sigma[k*D];   // Vector variance of cluster k
                // Responsibility of cluster k for data point i
                gamma[i*K + k] = pi[k] * gaussian_multi_diag(x, mu_k, sigma_k, D);
                // Accumulate denominator for normalization
                denom += gamma[i*K + k];
            }
            // Guard to avoid division by zero
            if (denom == 0.0 || isnan(denom)) denom = EPS_VAR;
            // Normalize responsibilities
            for (int k = 0; k < K; k++) gamma[i*K + k] /= denom;
        }

        // M-step
        // Reset accumulators
        memset(N_k, 0, (size_t)K * sizeof(double));
        memset(mu_k, 0, (size_t)K * D * sizeof(double));
        memset(sigma_k, 0, (size_t)K * D * sizeof(double));
        
        // Accumulate Nk and mu_num for each cluster
        for (int i = 0; i < N; i++) {
            double *x = &X[i*D]; // Vector of features for data point i
            for (int k = 0; k < K; k++) {
                N_k[k] += gamma[i*K + k]; // Accumulate responsibilities of a data point to cluster k
                for (int d = 0; d < D; d++) { 
                    // Weight the data point by its responsibility and accumulate for mean
                    mu_k[k*D + d] += gamma[i*K + k] * x[d];
                }
            }
        }

        // Finalize the calculation of the weighted means (for each feature) for each cluster
        for (int k = 0; k<K; k++) {
            // Guard to avoid division by zero
            if (N_k[k] <= 0.0) N_k[k] = EPS_VAR;
            // Finalize mu
            for( int d = 0; d < D; d++) {
                mu[k*D + d] = mu_k[k*D +d] / N_k[k];
            }
        }

        // Accumulate weighted squared differences for variances
        for (int i = 0; i < N; i++) {
            double *x = &X[i * D]; // Vector of features for data point i
            for (int k = 0; k < K; k++) {
                // For each feature dimension compute (x - mu)^2 and weight it by the responsibility
                for (int d = 0; d < D; d++) {
                    double diff = x[d] - mu[k * D + d];
                    sigma_k[k * D + d] += gamma[i * K + k] * diff * diff; // accumulate weighted squared difference contribution for cluster k
                }
            }
        }

        // Finalize sigma (variance per-dim) and pi
        for (int k = 0; k < K; k++) {
            for (int d = 0; d < D; d++) {
                // Nk[k] is already guarded
                // Finalize variance for each dimension
                sigma[k * D + d] = sigma_k[k * D + d] / N_k[k];
            }
            // Update mixture weights
            pi[k] = N_k[k] / (double)N;
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
        printf("Cluster %d: pi=%.6f\n", k, pi[k]);
        printf("  mu: ");
        for (int d = 0; d < D; d++) printf("%.6f ", mu[k * D + d]);
        printf("\n  sigma (std per-dim): ");
        for (int d = 0; d < D; d++) printf("%.6f ", sqrt(sigma[k * D + d]));
        printf("\n");
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
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&gamma,&N_k,&mu_k,&sigma_k);
    return 0;
}
