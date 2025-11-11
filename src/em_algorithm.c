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
 *     -N: number of samples
 *     -D: number of features
 *     -K: number of clusters
 *     Output parameters:
 *     -mu: (K x D) Matrix of cluster means, randomly initialized
 *     -sigma: (K x D) Matrix of cluster variances, initialized to 1
 *     -pi: (K) Vector of mixture weights, initialized to 1/K
 */
void init_params(double *X, int N, int D, int K, double *mu, double *sigma, double *pi){
    // Seed random number generator for reproducibility
    srand(0); 
    for (int k = 0; k < K; k++) { 
        // Choose a random data point index
        int r = rand() % N;
        // For each feature of the data point
        for (int d = 0; d < D; d++) {
            // Initialize mean of cluster k, feature d to the value of the randomly selected data point
            mu[k*D + d] = X[r*D + d];
            // Initialize variances to 1.0
            sigma[k*D + d] = 1.0;
        }
        // Initialize mixture weights uniformly (each cluster has equal weight 1 / K)
        pi[k] = 1.0 / (double) K;
    }
}

/**
 *  Compute predicted labels from responsibilities (gamma)
 *   Parameters:
 *     - gamma: (N x K) Responsibilities matrix
 *     - N: Number of samples
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
 *    - N: number of samples
 *    - D: number of features
 *    - K: number of clusters
 *    - mu: (K x D) Matrix of cluster means
 *    - sigma: (K x D) Matrix of cluster variances
 *    - pi: (K) Vector of mixture weights
 *    Output parameters:
 *    - gamma: (N x K) Responsibilities matrix
*/
void e_step(double *X, int N, int D, int K, double *mu, double *sigma, double *pi, double *gamma){
    for(int i = 0; i < N; i++) {
        // Initialize denominator for normalization
        double denom = 0.0;
        // Pointer to the i-th data point
        double *x = &X[i*D];
        for (int k = 0; k < K; k++) {
            double *mu_k = &mu[k*D];         // Vector mean of cluster k
            double *sigma_k = &sigma[k*D];   // Vector variance of cluster k
        
            // Responsibility of cluster k for data point i
            gamma[i*K + k] = pi[k] * gaussian_multi_diag(x, mu_k, sigma_k, D);
            // Accumulate denominator for normalization
            denom += gamma[i*K + k];
        }
        // Guard to avoid division by zero
        if (denom == 0.0 || isnan(denom)) denom = GUARD_VALUE;
        // Normalize responsibilities
        for (int k = 0; k < K; k++) gamma[i*K + k] /= denom;
    }
}

/**
 *  M-step: Update parameters (mu, sigma, pi) for each cluster
 *   Parameters:
 *    - X: dataset (N x D)
 *    - N: number of samples
 *    - D: number of features
 *    - K: number of clusters
 *    - gamma: (N x K) Responsibilities matrix
 *    - N_k: (K) Sum of responsibilities per cluster 
 *    - mu_k: (K x D) Weighted sums for means 
 *    - sigma_k: (K x D) Weighted sums for variances 
 *    Output parameters:
 *    - mu: (K x D) Matrix of cluster means
 *    - sigma: (K x D) Matrix of cluster variances
 *    - pi: (K) Vector of mixture weights

*/
void m_step( double *X, int N, int D, int K, double *gamma, double *mu, double *sigma, double *pi, double *N_k, double *mu_k, double *sigma_k){
    reset_accumulators(N_k, mu_k, sigma_k, K, D);
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
        if (N_k[k] <= 0.0) N_k[k] = GUARD_VALUE;
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
                // Accumulate weighted squared difference
                sigma_k[k * D + d] += gamma[i * K + k] * diff * diff; 
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

/**
 *  M-step: Update parameters (mu, sigma, pi) for each cluster
 *   Parameters:
 *    - local_X: dataset (local_N x D)
 *    - N: number of samples
 *   - local_N: number of samples in the local partition
 *    - D: number of features
 *    - K: number of clusters
 *    - local_gamma: (local_N x K) Responsibilities matrix
 *    - N_k: (K) Sum of responsibilities per cluster 
 *    - mu_k: (K x D) Weighted sums for means 
 *    - sigma_k: (K x D) Weighted sums for variances 
 *    Output parameters:
 *    - mu: (K x D) Matrix of cluster means
 *    - sigma: (K x D) Matrix of cluster variances
 *    - pi: (K) Vector of mixture weights
 *    - rank: Id of the current process
*/
void m_step_parallelized(double *local_X, int N, int local_N, int D, int K, double *local_gamma, double *mu, double *sigma, double *pi, double *N_k, double *local_N_k, double *mu_k, double *local_mu_k,double *sigma_k, double *local_sigma_k, int rank){
    parallel_reset_accumulators(N_k, mu_k, sigma_k, local_N_k, local_mu_k, local_sigma_k, K, D);

    // Accumulate Nk and mu_num for each cluster, done by every process
    for (int i = 0; i < local_N; i++) {
        double *x = &local_X[i*D]; // Vector of features for data point i
        for (int k = 0; k < K; k++) {
            local_N_k[k] += local_gamma[i*K + k]; // Accumulate responsibilities of a data point to cluster k
            for (int d = 0; d < D; d++) { 
                // Weight the data point by its responsibility and accumulate for mean
                local_mu_k[k*D + d] += local_gamma[i*K + k] * x[d];
            }
        }
    }

    MPI_Reduce(local_N_k, N_k, K, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_mu_k, mu_k, D*K, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Finalize the calculation of the weighted means (for each feature) for each cluster, only done by rank 0
    if(rank == 0){
        //TODO: (IN FUTURE, should be done using openmp???)
        for (int k = 0; k < K; k++) {
            // Guard to avoid division by zero
            if (N_k[k] <= 0.0) N_k[k] = GUARD_VALUE;
            // Finalize mu
            for( int d = 0; d < D; d++) {
                mu[k*D + d] = mu_k[k*D +d] / N_k[k];
            }
        }
    }

    MPI_Bcast(mu, K*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Accumulate weighted squared differences for variances, done by every process
    for (int i = 0; i < local_N; i++) {
        double *x = &local_X[i * D]; // Vector of features for data point i
        for (int k = 0; k < K; k++) {
            // For each feature dimension compute (x - mu)^2 and weight it by the responsibility
            for (int d = 0; d < D; d++) {
                double diff = x[d] - mu[k * D + d];
                // Accumulate weighted squared difference
                local_sigma_k[k * D + d] += local_gamma[i * K + k] * diff * diff; 
            }
        }
    }
    
    MPI_Reduce(local_sigma_k, sigma_k, K*D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Finalize sigma (variance per-dim) and pi, only done by rank 0
    if(rank == 0){
        //TODO: (IN FUTURE, should be done using openmp???)
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

    MPI_Bcast(sigma, K*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pi, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}