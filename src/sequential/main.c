#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 500000     // number of data points
#define K 2       // number of clusters
#define MAX_ITER 100
#define EPS 1e-6

/* 
    Parameters and data structures needed for GMM
    - X[N]: vector of N data points
    - mu[K]: means of K Gaussian components that generated the data. Need to be estimated.
    - sigma[K]: variances of K Gaussian components. Need to be estimated.
    - pi[K]: mixture weights of K components. Represent the prior probability of each cluster
    - gamma[N][K]: responsibility matrix, where gamma[i][k] is the probability that data point i belongs to cluster k
*/
double X[N]; // data points
double mu[K], sigma[K], pi[K], resp[N][K];


double gaussian(double x, double mean, double var) {
    return (1.0 / sqrt(2 * M_PI * var)) * exp(- (x - mean)*(x - mean) / (2 * var));
}

int main() {
    // --- Generate sample data (for simplicity) ---
    for (int i = 0; i < N; i++) {
        X[i] = (i < N/4) ? (rand() % 100) / 10.0 : 5.0 + (rand() % 100) / 10.0;
    }

    // --- Initialize parameters ---
    for (int k = 0; k < K; k++) {
        mu[k] = X[rand() % N];
        sigma[k] = 1.0;
        pi[k] = 1.0 / K;
    }

    // --- EM loop ---
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // E-step
        for (int i = 0; i < N; i++) {
            double denom = 0.0;
            for (int k = 0; k < K; k++) {
                resp[i][k] = pi[k] * gaussian(X[i], mu[k], sigma[k]);
                denom += resp[i][k];
            }
            for (int k = 0; k < K; k++) resp[i][k] /= denom;
        }

        // M-step
        for (int k = 0; k < K; k++) {
            double Nk = 0.0, mu_num = 0.0, var_num = 0.0;
            for (int i = 0; i < N; i++) {
                Nk += resp[i][k];
                mu_num += resp[i][k] * X[i];
            }
            mu[k] = mu_num / Nk;
            for (int i = 0; i < N; i++) {
                var_num += resp[i][k] * (X[i] - mu[k]) * (X[i] - mu[k]);
            }
            sigma[k] = var_num / Nk;
            pi[k] = Nk / N;
        }

        // --- Clustering assignment ---
        int cluster[N];
        int cluster_count[K] = {0};

        for (int i = 0; i < N; i++) {
            int best_k = 0;
            double best_val = resp[i][0];
            for (int k = 1; k < K; k++) {
                if (resp[i][k] > best_val) {
                    best_val = resp[i][k];
                    best_k = k;
                }
            }
            cluster[i] = best_k;
            cluster_count[best_k]++;
        }

        // Summary
        for (int k = 0; k < K; k++)
            printf("Cluster %d points = %d\n", k, cluster_count[k]);
    }

    // --- Print final parameters ---
    for (int k = 0; k < K; k++) {
        printf("Cluster %d: mu=%.3f sigma=%.3f pi=%.3f\n", k, mu[k], sqrt(sigma[k]), pi[k]);
    }




    return 0;
}
