#include "headers/em_algorithm.h"

/*
    Multivariate Gaussian probability density function with diagonal covariance matrix.
    p(x | mu, Sigma_diag) =
            exp( -0.5 * ( D*log(2*pi) 
                         + sum_d log(sigma[d])
                         + sum_d (x[d]-mu[d])^2 / sigma[d] ) )
*/
//TODO: try OPENMP parallelization here
inline double gaussian_multi_diag(double *x, double *mu, double *sigma, int D) {
    double logdet = 0.0;
    double quad = 0.0;
    for (int d = 0; d < D; d++) {
        double var_d = sigma[d];
        // Guard to avoid division by zero
        if (var_d <= 0.0) var_d = GUARD_VALUE;
        logdet += log(var_d);
        double diff = x[d] - mu[d];
        quad += (diff * diff) / var_d;
    }
    double exponent = -0.5 * ( D * LOG_2PI + logdet + quad );
    return exp(exponent);
}

/**
 *   Initialize the parameters mu, sigma, pi for the EM algorithm:
 *    Parameters:
 *     -X: dataset (N x D)
 *     -metadata:
 *       -N: number of samples
 *       -D: number of features
 *       -K: number of clusters
 *     Output parameters:
 *      cluster_params:
 *       -mu: (K x D) Matrix of cluster means, randomly initialized
 *       -sigma: (K x D) Matrix of cluster variances, initialized to 1
 *       -pi: (K) Vector of mixture weights, initialized to 1/K
 */
void init_params(double *X, Metadata *metadata, ClusterParams *cluster_params) {
    // Seed random number generator for reproducibility
    srand(0); 
    for (int k = 0; k < metadata->K; k++) { 
        // Choose a random data point index
        int r = rand() % metadata->N;
        // For each feature of the data point
        for (int d = 0; d < metadata->D; d++) {
            // Initialize mean of cluster k, feature d to the value of the randomly selected data point
            cluster_params->mu[k*metadata->D + d] = X[r*metadata->D + d];
            // Initialize variances to 1.0
            cluster_params->sigma[k*metadata->D + d] = 1.0;
        }
        // Initialize mixture weights uniformly (each cluster has equal weight 1 / K)
        cluster_params->pi[k] = 1.0 / (double) metadata->K;
    }
}

/**
 *  Compute predicted labels from responsibilities (gamma)
 *   Parameters:
 *     - gamma: (N x K) Responsibilities matrix
 *     - N: Number of samples   (Taken separated in order to be run both to sequential and parallel)
 *     - K: Number of clusters
 *    Output parameters:
 *     - predicted_labels: (N) Output array for predicted labels
 */
void compute_clustering(double *gamma, int N, int K, int *predicted_labels) {
    for (int i = 0; i < N; i++) {
        // Initialize max_resp to the responsibility of the first cluster
        double max_resp = gamma[i * K];
        int max_k = 0;
        
        // Find the cluster with the maximum responsibility for data point i
        for (int k = 1; k < K; k++) {
            if (gamma[i * K + k] > max_resp) {
                max_resp = gamma[i * K + k];
                max_k = k;
            }
        }
        // Assign the predicted label as the cluster with the highest responsibility
        predicted_labels[i] = max_k;
    }
}

/**
 *   E-step: Compute responsibilities (gamma)
 *   Parameters:
 *    - X: dataset (N x D)
 *    - N: number of samples (Considered separated in order to work both in sequential and parallel)
 *    - metadata:
 *         - D: number of features
 *         - K: number of clusters
 *   - cluster_params: ClusterParams structure containing mu, sigma, and pi
 *         - mu: (K x D) Matrix of cluster means
 *         - sigma: (K x D) Matrix of cluster variances
 *         - pi: (K) Vector of mixture weights
 *    Output parameters:
 *    - gamma: (N x K) Responsibilities matrix
*/
void e_step(double *X, int N, Metadata *metadata, ClusterParams *cluster_params, double *gamma){
    for(int i = 0; i < N; i++) {
        // Initialize denominator for normalization
        double denom = 0.0;
        // Pointer to the i-th data point
        double *x = &X[i*metadata->D];
        for (int k = 0; k < metadata->K; k++) {
            double *mu_k = &cluster_params->mu[k*metadata->D];         // Vector mean of cluster k
            double *sigma_k = &cluster_params->sigma[k*metadata->D];   // Vector variance of cluster k
            // Responsibility of cluster k for data point i
            gamma[i*metadata->K + k] = cluster_params->pi[k] * gaussian_multi_diag(x, mu_k, sigma_k, metadata->D);
            // Accumulate denominator for normalization
            denom += gamma[i*metadata->K + k];
        }
        // Guard to avoid division by zero
        if (denom == 0.0 || isnan(denom)) denom = GUARD_VALUE;
        // Normalize responsibilities
        for (int k = 0; k < metadata->K; k++) gamma[i*metadata->K + k] /= denom;
    }
}

/**
 *  M-step: Update parameters (mu, sigma, pi) for each cluster
 *   Parameters:
 *   - X: dataset (N x D)
 *   - metadata:
 *          - N: number of samples
 *          - D: number of features
 *          - K: number of clusters
 *    - acc (accumulators):
 *          - N_k: (K) Sum of responsibilities per cluster 
 *          - mu_k: (K x D) Weighted sums for means 
 *          - sigma_k: (K x D) Weighted sums for variances 
 *    - gamma: (N x K) Responsibilities matrix
 *    Output parameters:
 *    -cluster_params:
 *          - mu: (K x D) Matrix of cluster means
 *          - sigma: (K x D) Matrix of cluster variances
 *          - pi: (K) Vector of mixture weights
 * 
*/
void m_step( double *X, Metadata *metadata, ClusterParams *cluster_params, Accumulators *acc, double *gamma){
    reset_accumulators(acc, metadata);
    // Accumulate Nk and mu_num for each cluster
    for (int i = 0; i < metadata->N; i++) {
        double *x = &X[i*metadata->D]; // Vector of features for data point i
        for (int k = 0; k < metadata->K; k++) {
            acc->N_k[k] += gamma[i*metadata->K + k]; // Accumulate responsibilities of a data point to cluster k
            for (int d = 0; d < metadata->D; d++) { 
                // Weight the data point by its responsibility and accumulate for mean
                acc->mu_k[k*metadata->D + d] += gamma[i*metadata->K + k] * x[d];
            }
        }
    }

    // Finalize the calculation of the weighted means (for each feature) for each cluster
    for (int k = 0; k<metadata->K; k++) {
        // Guard to avoid division by zero
        if (acc->N_k[k] <= 0.0) acc->N_k[k] = GUARD_VALUE;
        // Finalize mu
        for( int d = 0; d < metadata->D; d++) {
            cluster_params->mu[k*metadata->D + d] = acc->mu_k[k*metadata->D +d] / acc->N_k[k];
        }
    }

    // Accumulate weighted squared differences for variances
    for (int i = 0; i < metadata->N; i++) {
        double *x = &X[i * metadata->D]; // Vector of features for data point i
        for (int k = 0; k < metadata->K; k++) {
            // For each feature dimension compute (x - mu)^2 and weight it by the responsibility
            for (int d = 0; d < metadata->D; d++) {
                double diff = x[d] - cluster_params->mu[k * metadata->D + d];
                // Accumulate weighted squared difference
                acc->sigma_k[k * metadata->D + d] += gamma[i * metadata->K + k] * diff * diff; 
            }
        }
    }

    // Finalize sigma (variance per-dim) and pi
    for (int k = 0; k < metadata->K; k++) {
        for (int d = 0; d < metadata->D; d++) {
            // Nk[k] is already guarded
            // Finalize variance for each dimension
            cluster_params->sigma[k * metadata->D + d] = acc->sigma_k[k * metadata->D + d] / acc->N_k[k];
        }
        // Update mixture weights
        cluster_params->pi[k] = acc->N_k[k] / (double)metadata->N;
    }
}

/**
 *  Parallel M-step: Update parameters (mu, sigma, pi) for each cluster
 *   Parameters:
 *   - local_X: local dataset chunk (local_N x D)
 *   - local_N: number of samples in the local chunk
 *   - metadata:
 *          - N: number of samples
 *          - D: number of features
 *          - K: number of clusters
 *    - cluster_params: ClusterParams structure containing mu, sigma, and pi
 *          - mu: (K x D) Matrix of cluster means
 *          - sigma: (K x D) Matrix of cluster variances
 *          - pi: (K) Vector of mixture weights
 *    - cluster_acc (global accumulators):
 *          - N_k: (K) Sum of responsibilities per cluster 
 *          - mu_k: (K x D) Weighted sums for means 
 *          - sigma_k: (K x D) Weighted sums for variances 
 *    - local_cluster_acc (local accumulators):
 *          - N_k: (K) Sum of responsibilities per cluster 
 *          - mu_k: (K x D) Weighted sums for means 
 *          - sigma_k: (K x D) Weighted sums for variances 
 *    - local_gamma: (local_N x K) Local responsibilities matrix
 *    - rank: MPI rank of the current process
 * 
*/
void m_step_parallelized(double *local_X, int local_N, Metadata *metadata, ClusterParams *cluster_params, Accumulators *cluster_acc, Accumulators *local_cluster_acc, double *local_gamma, int rank){
    parallel_reset_accumulators(cluster_acc, local_cluster_acc, metadata);

    // Accumulate Nk and mu_num for each cluster, done by every process
    for (int i = 0; i < local_N; i++) {
        double *x = &local_X[i*metadata->D]; // Vector of features for data point i
        for (int k = 0; k < metadata->K; k++) {
            local_cluster_acc->N_k[k] += local_gamma[i*metadata->K + k]; // Accumulate responsibilities of a data point to cluster k
            for (int d = 0; d < metadata->D; d++) { 
                // Weight the data point by its responsibility and accumulate for mean
                local_cluster_acc->mu_k[k*metadata->D + d] += local_gamma[i*metadata->K + k] * x[d];
            }
        }
    }

    MPI_Reduce(local_cluster_acc->N_k, cluster_acc->N_k, metadata->K, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_cluster_acc->mu_k, cluster_acc->mu_k, metadata->D*metadata->K, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Finalize the calculation of the weighted means (for each feature) for each cluster, only done by rank 0
    if(rank == 0){
        //TODO: (IN FUTURE, should be done using openmp???)
        for (int k = 0; k < metadata->K; k++) {
            // Guard to avoid division by zero
            if (cluster_acc->N_k[k] <= 0.0) cluster_acc->N_k[k] = GUARD_VALUE;
            // Finalize mu
            for( int d = 0; d < metadata->D; d++) {
               cluster_params->mu[k*metadata->D + d] = cluster_acc->mu_k[k*metadata->D +d] / cluster_acc->N_k[k];
            }
        }
    }

    MPI_Bcast(cluster_params->mu, metadata->K*metadata->D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Accumulate weighted squared differences for variances, done by every process
    for (int i = 0; i < local_N; i++) {
        double *x = &local_X[i * metadata->D]; // Vector of features for data point i
        for (int k = 0; k < metadata->K; k++) {
            // For each feature dimension compute (x - mu)^2 and weight it by the responsibility
            for (int d = 0; d < metadata->D; d++) {
                double diff = x[d] - cluster_params->mu[k * metadata->D + d];
                // Accumulate weighted squared difference
                local_cluster_acc->sigma_k[k * metadata->D + d] += local_gamma[i * metadata->K + k] * diff * diff; 
            }
        }
    }
    
    MPI_Reduce(local_cluster_acc->sigma_k, cluster_acc->sigma_k, metadata->K*metadata->D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Finalize sigma (variance per-dim) and pi, only done by rank 0
    if(rank == 0){
        //TODO: (IN FUTURE, should be done using openmp???)
        for (int k = 0; k < metadata->K; k++) {
            for (int d = 0; d < metadata->D; d++) {
                // Nk[k] is already guarded
                // Finalize variance for each dimension
                cluster_params->sigma[k * metadata->D + d] = cluster_acc->sigma_k[k * metadata->D + d] / cluster_acc->N_k[k];
            }
            // Update mixture weights
            cluster_params->pi[k] = cluster_acc->N_k[k] / (double)metadata->N;
        }
    }

    MPI_Bcast(cluster_params->sigma, metadata->K*metadata->D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(cluster_params->pi, metadata->K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}