#include "headers/main.h"

#define MAX_ITER 100

/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file> <execution_info_file> [output_labels_file]
*/
int main(int argc, char **argv) { 
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Id of the current process
    int rank;
    // Number of processes
    int size;
    
    // Initialize MPI rank (process ID) and size (total processes)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Metadata 
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    int max_line_size;                  // Maximum number of character in a line in the dataset file
    
    // Global arrays
    double *X = NULL;                   // X[N * D] Vector of data points
    double *mu = NULL;                  // mu[k * D] Vector of feature means per cluster; mu[k * D + d] is the mean of feature d for cluster k
    double *sigma = NULL;               // sigma[k * D] Vector of variance per each feature per cluster; sigma[k * D + d] is the variance of feature d for cluster k
    double *pi = NULL;                  // pi[k] Vector of mixture weights: prior probability that a random data point belongs to cluster k
    
    // Global accumulators for m-step
    double *N_k = NULL;                 // N_k[k] = Sum of responsibilities per cluster
    double *mu_k = NULL;                // mu_k[k * D] = Weighted sums for means
    double *sigma_k = NULL;             // sigma_k[k * D] = Weighted sums for variances

    // Local variables for parallel computation
    double *local_X = NULL;             // Local data points for this MPI process (local_N * D)
    double *local_gamma = NULL;         // local_gamma[local_N * K] Local responsibilities vector for each process. local_gamma[i * K + k] is the responsibility of cluster k for data point i
    double *local_N_k = NULL;           // Local version of N_k for parallelizing
    double *local_mu_k = NULL;          // Local version of mu_k for parallelizing
    double *local_sigma_k = NULL;       // Local version of sigma_k for parallelizing

    // Label vectors for clustering
    int *predicted_labels = NULL;       // Predicted cluster labels
    int *local_predicted_labels = NULL; // Local predicted cluster labels
    int *ground_truth_labels = NULL;    // Ground truth labels

    // Initializing all times
    double start_time = MPI_Wtime();    
    double io_time = 0.0, compute_time = 0.0, e_step_time = 0.0, m_step_time = 0.0, data_distribution_time = 0.0;

    // Check command line arguments
    if(argc < 4 || argc > 5){
        if (rank == 0) fprintf(stderr, "Usage: %s <dataset_file> <metadata_file> <execution_info_file> [output_labels_file]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    if(argv[1] == NULL || argv[2] == NULL || argv[3] == NULL || (argc > 4 && argv[4] == NULL)){
        if (rank == 0) fprintf(stderr, "Dataset file, metadata and execution info file must be provided\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Get filenames from arguments
    const char *filename = argv[1];
    const char *metadata_filename = argv[2];
    const char *execution_info_filename = argv[3];
    const char *output_labels_file = (argc > 4) ? argv[4] : NULL;

    double io_start = MPI_Wtime();
    // Read metadata from metadata file (only by rank 0)
    if (rank == 0) {
        int meta_status = read_metadata(metadata_filename, &N, &D, &K, &max_line_size);
        if(meta_status != 0){
            fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        printf("Metadata: samples N=%d, features D=%d, clusters K=%d\n", N, D, K);
    }
    io_time += MPI_Wtime() - io_start;

    // Broadcast N, D, K to all process using a single MPI call
    double data_distribution_start = MPI_Wtime();
    int metadata[3];
    if(rank == 0) {
        metadata[0] = N;
        metadata[1] = D;
        metadata[2] = K;
    }
    MPI_Bcast(metadata, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank!=0){
        N = metadata[0];
        D = metadata[1];
        K = metadata[2];
    }
    data_distribution_time += MPI_Wtime() - data_distribution_start;
    
    int alloc_fail = 0;     // Flag to check correctness of all allocations
    // Allocate buffers for master process
    if(rank == 0){
        X = malloc(N * D * sizeof(double));
        predicted_labels = malloc(N * sizeof(int));
        ground_truth_labels = malloc(N * sizeof(int));
        if(!X || !predicted_labels || !ground_truth_labels) {
            alloc_fail = 1;
        }
    }
    // Allocate the buffers needed by all processes
    mu = malloc(K * D * sizeof(double)); 
    sigma = malloc(K * D * sizeof(double)); 
    pi = malloc(K * sizeof(double));
    N_k = malloc((size_t)K * sizeof(double));              
    mu_k = malloc((size_t)K * D * sizeof(double));   
    sigma_k = malloc((size_t)K * D * sizeof(double));  
    if(!mu || !sigma || !pi || !N_k || !mu_k || !sigma_k){
        alloc_fail = 1;
    }
    // Check that all allocations were successful
    if(alloc_fail){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&local_gamma,&N_k,&mu_k,&sigma_k);
        MPI_Abort(MPI_COMM_WORLD,1);
    }   

    // Read dataset (only by rank 0)
    io_start = MPI_Wtime();
    if(rank == 0){
        if(read_dataset(filename, D, N, max_line_size, X, ground_truth_labels) != 0){
            fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
            safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&local_gamma,&N_k,&mu_k,&sigma_k);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        // If correctly read, debug print first few samples
        debug_print_first_samples(N, D, X, ground_truth_labels);
    }
    io_time += MPI_Wtime() - io_start;
    
    //Initialize parameters for the EM algorithm (Done only by rank 0)
    if (rank == 0) init_params(X, N, D, K, mu, sigma, pi);
    
    // Broadcast initial parameters to all processes
    //TODO: check if this can be optimized by using a single MPI call using DERIVED datatype
    MPI_Bcast(mu, K * D,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(sigma, K * D,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pi, K,  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Prepare for scattering data
    data_distribution_start = MPI_Wtime();
    
    // Compute the number of local samples for each process
    int local_N = N / size;
    // Compute the remainder and distribute it
    if (rank < N % size) local_N++;

    // Allocate the vectors for local data
    local_X = malloc(local_N * D * sizeof(double));
    local_gamma = malloc(local_N * K * sizeof(double));
    local_predicted_labels = malloc(local_N * sizeof(int)); 
    local_N_k = malloc((size_t)K * sizeof(double));              
    local_mu_k = malloc((size_t)K * D * sizeof(double));   
    local_sigma_k = malloc((size_t)K * D * sizeof(double));

    // Check that all allocations were successful
    if (!local_X || !local_gamma || !local_predicted_labels || !local_N_k || !local_mu_k || !local_sigma_k) {
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&local_gamma,&N_k,&mu_k,&sigma_k);
        safe_cleanup_local(&local_N_k,&local_mu_k,&local_sigma_k);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Scatter dataset X from rank 0 to all processes; each gets local_N rows in local_X.
    scatter_dataset(X, local_X, N, local_N, D, rank, size);

    data_distribution_time += MPI_Wtime() - data_distribution_start;

    // Debug print local data distribution
    debug_print_scatter(local_N, D, local_X, rank);
    double compute_start = MPI_Wtime();
    
    /*
        EM loop
        The loop runs until MAX_ITER is reached
    */
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // E-step
        double e_step_start = MPI_Wtime();
        e_step(local_X, local_N, D, K, mu, sigma, pi, local_gamma);
        e_step_time += MPI_Wtime() - e_step_start;

        //M-step
        double m_step_start = MPI_Wtime();
        m_step_parallelized(local_X, N, local_N, D, K, local_gamma,  mu, sigma, pi, N_k, local_N_k, mu_k, local_mu_k, sigma_k, local_sigma_k, rank);
        m_step_time += MPI_Wtime() - m_step_start;
    }
    
    // Compute local predicted labels from responsibilities
    compute_clustering(local_gamma, local_N, K, local_predicted_labels);
    compute_time += MPI_Wtime() - compute_start;

    // Gather predicted labels from all processes
    gather_dataset(local_predicted_labels, predicted_labels, N, local_N, rank, size);

    io_start = MPI_Wtime();
    if (rank == 0) {
        // Print final parameters 
        debug_print_cluster_params(K, D, mu, sigma, pi);
        // Write final cluster assignments to file to validate
        if (output_labels_file){
            int write_status = write_labels_info(output_labels_file, predicted_labels, ground_truth_labels, N);
            if(write_status != 0){
                fprintf(stderr, "Failed to write labels to file: %s\n", output_labels_file);
            }
        }
    }
    io_time += MPI_Wtime() - io_start;

    // Report the execution info (only by rank 0)
    if(rank == 0){
        if(write_execution_info(execution_info_filename, size, N, D, K, MPI_Wtime() - start_time, io_time, compute_time, e_step_time, m_step_time, data_distribution_time) != 0){
            fprintf(stderr, "Failed to write execution info to file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Free all the allocated memory
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&local_gamma,&N_k,&mu_k,&sigma_k);
    safe_cleanup_local(&local_N_k,&local_mu_k,&local_sigma_k);
    // Finalize MPI communicator
    MPI_Finalize();
    return 0;
}