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
    Allocates memory for cluster accumulators N_k, mu_k, sigma_k
*/
int alloc_accumulators(Accumulators *acc, Metadata *metadata) {
    acc->N_k = malloc(metadata->K * sizeof(double)); 
    acc->mu_k = malloc(metadata->K * metadata->D * sizeof(double)); 
    acc->sigma_k = malloc(metadata->K * metadata->D * sizeof(double));
    if (!acc->N_k || !acc->mu_k || !acc->sigma_k) {
        fprintf(stderr, "Memory allocation failed for cluster accumulators\n");
        return -1; // Allocation failed
    }
    return 0; // Success
}

/*
    Frees memory allocated for cluster accumulators N_k, mu_k, sigma_k
*/
void free_accumulators(Accumulators *acc) {
    free_and_null((void**)&acc->N_k);
    free_and_null((void**)&acc->mu_k);
    free_and_null((void**)&acc->sigma_k);
}

/*
    Safely frees all allocated memory (passed by address)
*/
void safe_cleanup(
    ClusterParams *cluster_params,
    Accumulators *cluster_acc,
    double **X,
    int **predicted_labels,
    int **ground_truth_labels,
    double **resp
)
{
    free_cluster_params(cluster_params);
    free_accumulators(cluster_acc);
    free_and_null((void**)X);
    free_and_null((void**)predicted_labels);
    free_and_null((void**)ground_truth_labels);
    free_and_null((void**)resp);
}

/*
    Reset accumulators used in the M-step of the EM algorithm
*/
void reset_accumulators(Accumulators *acc, Metadata *metadata) {
    memset(acc->N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(acc->mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(acc->sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
}

/*
    Reset accumulators used in the parallel M-step of the EM algorithm
*/
void parallel_reset_accumulators(Accumulators *acc, Accumulators *local_acc, Metadata *metadata) {
    memset(acc->N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(acc->mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(acc->sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(local_acc->N_k, 0, (size_t)metadata->K * sizeof(double));
    memset(local_acc->mu_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
    memset(local_acc->sigma_k, 0, (size_t)metadata->K * metadata->D * sizeof(double));
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
            - -d <fileName>: debug information file (optional)
            - -t <value>: convergence threshold (optional)
    Output:
        - inputParams: struct containing the parsed parameters
    Returns:
        - 0 on success, -1 on failure
*/
int parseParameter(int argc, char **argv, InputParams_t *inputParams) {
    int opt;
    // Initialize all parameters to NULL
    inputParams->dataset_file_path = NULL;
    inputParams->meta_data_file_path = NULL;
    inputParams->benchmarks_file_path = NULL;
    inputParams->output_file_path = NULL;
    inputParams->debug_file_path = NULL;
    inputParams->threshold = 0.0;

    while ((opt = getopt(argc, argv, "i:m:b:o:d:t:")) != -1) {
        switch (opt) {
        case 'i':
            inputParams->dataset_file_path = optarg;
            break;
        case 'm':
            inputParams->meta_data_file_path = optarg;
            break;
        case 'b':
            inputParams->benchmarks_file_path = optarg;
            break;
        case 'o':
            inputParams->output_file_path = optarg;
            break;
        case 'd':
            inputParams->debug_file_path = optarg;
            break;
        case 't':
            inputParams->threshold = atof(optarg);
            break;
        // If an unknown option is provided print the usage
        default:
            fprintf(stderr, "Usage: %s -i input_file -m metadata_file -b benchmarks_file -o output_file -d debug_file -t threshold\n", argv[0]);
            return -1;
        }
    }
    // Check for required parameters
    if (inputParams->dataset_file_path == NULL || inputParams->meta_data_file_path == NULL) {
        fprintf(stderr, "Usage: %s -i input_file -m metadata_file -b benchmarks_file -o output_file\n", argv[0]);
        return -1;
    }
    return 0;
}