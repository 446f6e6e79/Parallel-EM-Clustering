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
    Allocates memory for cluster parameters mu, sigma, pi
*/
int alloc_cluster_params(ClusterParams *params, Metadata *metadata) {
    params->mu = malloc(metadata->K * metadata->D * sizeof(double)); 
    params->sigma = malloc(metadata->K * metadata->D * sizeof(double)); 
    params->pi = malloc(metadata->K * sizeof(double));
    if (!params->mu || !params->sigma || !params->pi) {
        fprintf(stderr, "Memory allocation failed for cluster parameters\n");
        return -1; // Allocation failed
    }
    return 0; // Success
}

/*
    Frees memory allocated for cluster parameters mu, sigma, pi
*/
void free_cluster_params(ClusterParams *params) {
    free_and_null((void**)&params->mu);
    free_and_null((void**)&params->sigma);
    free_and_null((void**)&params->pi);
}

/*
    Safely frees all allocated memory (passed by address)
*/
void safe_cleanup(
    double **X,
    int **predicted_labels,
    int **ground_truth_labels,
    ClusterParams *cluster_params,
    double **resp,
    double **N_k,
    double **mu_k,
    double **sigma_k
)
{
    free_and_null((void**)X);
    free_and_null((void**)predicted_labels);
    free_and_null((void**)ground_truth_labels);
    free_cluster_params(cluster_params);
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
void reset_accumulators(double *N_k, double *mu_k, double *sigma_k, Metadata *metadata) {
    memset(N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
}

/*
    Reset accumulators used in the parallel M-step of the EM algorithm
*/
void parallel_reset_accumulators(double *N_k, double *mu_k, double *sigma_k, double *local_N_k, double *local_mu_k, double *local_sigma_k, Metadata *metadata) {
    memset(N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(local_N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(local_mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(local_sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
}

/*
    Saves in t the start time
*/
void start_timer(double *t) {
    *t = MPI_Wtime();
}
/*
    Stops the timer and accumulates the elapsed time in accumulator
*/
void stop_timer(double *t, double *accumulator) {
    *accumulator += MPI_Wtime() - *t;
}
/*
    Initializes all timers accumulator in the Timers struct to zero
*/
void initialize_timers(Timers_t *timers) {
    timers->total_time = 0.0;
    timers->io_time = 0.0;
    timers->compute_time = 0.0;
    timers->e_step_time = 0.0;
    timers->m_step_time = 0.0;
    timers->data_distribution_time = 0.0;
}

/*
    Parse the parameters of the program.
    Parameters:
        - argc: number of command line arguments
        - argv: array of command line arguments. The allowed arguments are:
            - -i <fileName>: input dataset file (required)
            - -m <fileName>: metadata file (required)
            - -b <fileName>: benchmarks output file (optional)
            - -o <fileName>: predicted labels output file (optional)
    Output:
        - inputParams: struct containing the parsed parameters
    Returns:
        - 0 on success, -1 on failure
*/
int parseParameter(int argc, char **argv, InputParams_t *inputParams) {
    int opt;
    while ((opt = getopt(argc, argv, "i:m:b:o:")) != -1) {
        switch (opt) {
        case 'i':
            inputParams->dataset_filename = optarg;
            break;
        case 'm':
            inputParams->metadata_filename = optarg;
            break;
        case 'b':
            inputParams->benchmarks_filename = optarg;
            break;
        case 'o':
            inputParams->output_filename = optarg;
            break;
        // If an unknown option is provided print the usage
        default:
            fprintf(stderr, "Usage: %s -i input_file -m metadata_file -b benchmarks_file -o output_file\n", argv[0]);
            return -1;
        }
    }
    // Check for required parameters
    if (inputParams->dataset_filename == NULL || inputParams->metadata_filename == NULL) {
        fprintf(stderr, "Usage: %s -i input_file -m metadata_file -b benchmarks_file -o output_file\n", argv[0]);
        return -1;
    }
    return 0;
}