#include <headers/utils.h>

/*
    Helper function to free a pointer and set it to NULL
*/
static inline void free_and_null(void **p) {
    if (p && *p) {
        free(*p);
        *p = NULL;
    }
}

/*
    Safely frees all allocated memory (passed by address)
*/
void safe_cleanup(
    double **X,
    int **predicted_labels,
    int **ground_truth_labels,
    double **mu,
    double **sigma,
    double **pi,
    double **resp,
    double **N_k,
    double **mu_k,
    double **sigma_k
)
{
    free_and_null((void**)X);
    free_and_null((void**)predicted_labels);
    free_and_null((void**)ground_truth_labels);
    free_and_null((void**)mu);
    free_and_null((void**)sigma);
    free_and_null((void**)pi);
    free_and_null((void**)resp);
    free_and_null((void**)N_k);
    free_and_null((void**)mu_k);
    free_and_null((void**)sigma_k);
}

/*
    Safely frees all allocated memory (passed by address)
    This version is made up for the parallel version
*/
void safe_cleanup_local(
    double **local_N_k,
    double **local_mu_k,
    double **local_sigma_k
)
{
    free_and_null((void**)local_N_k);
    free_and_null((void**)local_mu_k);
    free_and_null((void**)local_sigma_k);
}

/*
    Reset accumulators used in the M-step of the EM algorithm
*/
void reset_accumulators(double *N_k, double *mu_k, double *sigma_k, int K, int D) {
    memset(N_k, 0, (size_t)K * sizeof(double));
    memset(mu_k, 0, (size_t)K * D * sizeof(double));
    memset(sigma_k, 0, (size_t)K * D * sizeof(double));
}

/*
    Reset accumulators used in the parallel M-step of the EM algorithm
*/
void parallel_reset_accumulators(double *N_k, double *mu_k, double *sigma_k, double *local_N_k, double *local_mu_k, double *local_sigma_k, int K, int D) {
    memset(N_k, 0, (size_t)K * sizeof(double));
    memset(mu_k, 0, (size_t)K * D * sizeof(double));
    memset(sigma_k, 0, (size_t)K * D * sizeof(double));
    memset(local_N_k, 0, (size_t)K * sizeof(double));
    memset(local_mu_k, 0, (size_t)K * D * sizeof(double));
    memset(local_sigma_k, 0, (size_t)K * D * sizeof(double));
}
