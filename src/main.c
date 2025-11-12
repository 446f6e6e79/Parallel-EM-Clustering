#include "headers/main.h"

#define MAX_ITER 100

/*
    Expection-Maximization Clustering Algorithm
    Usage: ./program <dataset_file> <metadata_file> <execution_info_file> [output_labels_file]
*/
int main(int argc, char **argv) { 
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank;       // id of the current process
    int size;       // Number of processes
    
    // Initialize MPI rank (process ID) and size (total processes)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Metadata 
    Metadata metadata;                  // Contains number of samples, number of features, number of clusters and max line size
    
    // Global to all processes
    double *X = NULL;                   // X[N * D] Vector of data points
    ClusterParams cluster_params;       // Cluster parameters: mu (D * K), sigma (D * K), pi (K)

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

    // Initialize the timers
    Timers_t timers;
    initialize_timers(&timers);

    start_timer(&timers.start_time);
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
    
    // Read metadata from metadata file (only by rank 0)
    start_timer(&timers.io_start);
    if (rank == 0) {
        int meta_status = read_metadata(metadata_filename, &metadata);
        if(meta_status != 0){
            fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        printf("Metadata: samples N=%d, features D=%d, clusters K=%d\n", metadata.N, metadata.D, metadata.K);
    }
    stop_timer(&timers.io_start, &timers.io_time);


    // Broadcast N, D, K to all process using a single MPI call
    start_timer(&timers.data_distribution_start);
    broadcast_metadata(&metadata, rank);
    stop_timer(&timers.data_distribution_start, &timers.data_distribution_time);

    int alloc_fail = 0;     // Flag to check correctness of all allocations
    // Allocate buffers for master process
    if(rank == 0){
        X = malloc(metadata.N * metadata.D * sizeof(double));
        predicted_labels = malloc(metadata.N * sizeof(int));
        ground_truth_labels = malloc(metadata.N * sizeof(int));
        if(!X || !predicted_labels || !ground_truth_labels) {
            alloc_fail = 1;
        }
    }

    // Allocate the buffers needed by all processes
    int cluster_alloc_status = alloc_cluster_params(&cluster_params, &metadata);
    N_k = malloc((size_t)metadata.K * sizeof(double));              
    mu_k = malloc((size_t)metadata.K * metadata.D * sizeof(double));   
    sigma_k = malloc((size_t)metadata.K * metadata.D * sizeof(double));  
    if(cluster_alloc_status != 0 || !N_k || !mu_k || !sigma_k){
        alloc_fail = 1;
    }
    // Check that all allocations were successful
    if(alloc_fail){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&local_gamma,&N_k,&mu_k,&sigma_k);
        MPI_Abort(MPI_COMM_WORLD,1);
    }   

    // Read dataset (only by rank 0)
    start_timer(&timers.io_start);
    if(rank == 0){
        if(read_dataset(filename, &metadata, X, ground_truth_labels) != 0){
            fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
            safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&local_gamma,&N_k,&mu_k,&sigma_k);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    stop_timer(&timers.io_start, &timers.io_time);
    
    //Initialize parameters for the EM algorithm (Done only by rank 0)
    if (rank == 0) init_params(X, &metadata, &cluster_params);
    
    // Broadcast in a single time initial parameters to all processes
    start_timer(&timers.data_distribution_start);
    broadcast_clusters_parameters(cluster_params, &metadata);

    // Distribute data among processes
    int local_N = compute_local_N(metadata.N, size, rank);
    
    // Allocate the vectors for local data
    local_X = malloc(local_N * metadata.D * sizeof(double));
    local_gamma = malloc(local_N * metadata.K * sizeof(double));
    local_predicted_labels = malloc(local_N * sizeof(int)); 
    local_N_k = malloc((size_t)metadata.K * sizeof(double));              
    local_mu_k = malloc((size_t)metadata.K * metadata.D * sizeof(double));   
    local_sigma_k = malloc((size_t)metadata.K * metadata.D * sizeof(double));

    // Check that all allocations were successful
    if (!local_X || !local_gamma || !local_predicted_labels || !local_N_k || !local_mu_k || !local_sigma_k) {
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&local_gamma,&N_k,&mu_k,&sigma_k);
        safe_cleanup_local(&local_N_k,&local_mu_k,&local_sigma_k);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Scatter dataset X from rank 0 to all processes; each gets local_N rows in local_X.
    scatter_dataset(X, local_X, local_N, &metadata, rank, size);
    stop_timer(&timers.data_distribution_start, &timers.data_distribution_time);
    
    /*
        EM loop
        The loop runs until MAX_ITER is reached
    */
   start_timer(&timers.compute_start);
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // E-step
        start_timer(&timers.e_step_start);
        e_step(local_X, local_N, &metadata, &cluster_params, local_gamma);
        stop_timer(&timers.e_step_start, &timers.e_step_time);

        //M-step
        start_timer(&timers.m_step_start);
        m_step_parallelized(local_X, local_N, &metadata, local_gamma, &cluster_params, N_k, local_N_k, mu_k, local_mu_k, sigma_k, local_sigma_k, rank);
        stop_timer(&timers.m_step_start, &timers.m_step_time);
    }
    
    // Compute local predicted labels from responsibilities
    compute_clustering(local_gamma, local_N, metadata.K, local_predicted_labels);
    stop_timer(&timers.compute_start, &timers.compute_time);
    
    // Gather predicted labels from all processes
    gather_dataset(local_predicted_labels, predicted_labels, metadata.N, local_N, rank, size);

    // Write the execution output
    start_timer(&timers.io_start);
    if (rank == 0) {
        // Print final parameters 
        debug_print_cluster_params(&metadata, &cluster_params);
        // Write final cluster assignments to file to validate
        if (output_labels_file){
            int write_status = write_labels_info(output_labels_file, predicted_labels, ground_truth_labels, metadata.N);
            if(write_status != 0){
                fprintf(stderr, "Failed to write labels to file: %s\n", output_labels_file);
            }
        }
    }
    stop_timer(&timers.io_start, &timers.io_time);
    
    // Stop total execution timer
    stop_timer(&timers.start_time, &timers.total_time);
    // Report the execution info (only by rank 0)
    if(rank == 0){
        if(write_execution_info(execution_info_filename, size, &metadata, &timers) != 0){
            fprintf(stderr, "Failed to write execution info to file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Free all the allocated memory
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&local_gamma,&N_k,&mu_k,&sigma_k);
    safe_cleanup_local(&local_N_k,&local_mu_k,&local_sigma_k);
    // Finalize MPI communicator
    MPI_Finalize();
    return 0;
}