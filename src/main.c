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
    Metadata metadata;                      // Contains number of samples, number of features, number of clusters and max line size
    
    // Global to all processes
    double *X = NULL;                       // X[N * D] Vector of data points
    ClusterParams cluster_params;           // Contains cluster parameters: mu (D * K), sigma (D * K), pi (K)

    // Global accumulators for m-step
    Accumulators cluster_acc;        // Contains accumulators for m-step: N_k (K), mu_k (D * K), sigma_k (D * K)

    // Local variables for parallel computation
    double *local_X = NULL;                 // Local data points for this MPI process (local_N * D)
    double *local_gamma = NULL;             // local_gamma[local_N * K] Local responsibilities vector for each process. local_gamma[i * K + k] is the responsibility of cluster k for data point i
    Accumulators local_cluster_acc;  // Local accumulators for m-step: local_N_k (K), local_mu_k (D * K), local_sigma_k (D * K)

    // Label vectors for clustering
    int *predicted_labels = NULL;           // Predicted cluster labels
    int *local_predicted_labels = NULL;     // Local predicted cluster labels
    int *ground_truth_labels = NULL;        // Ground truth labels

    // Initialize the timers
    Timers_t timers;
    initialize_timers(&timers);

    // Input parameters
    InputParams_t inputParams;

    start_timer(&timers.start_time);
    // Parse input parameters
    if (parseParameter(argc, argv, &inputParams) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error parsing input parameters.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Read metadata from metadata file (only by rank 0)
    start_timer(&timers.io_start);
    if (rank == 0) {
        int meta_status = read_metadata(inputParams.metadata_filename, &metadata);
        if(meta_status != 0){
            fprintf(stderr, "Failed to read metadata from file: %s\n", inputParams.metadata_filename);
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
    int acc_alloc_status =  alloc_accumulators(&cluster_acc, &metadata); 
    if(cluster_alloc_status != 0 || acc_alloc_status != 0){
        alloc_fail = 1;
    }
    // Check that all allocations were successful
    if(alloc_fail){
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&cluster_acc,&local_gamma);
        MPI_Abort(MPI_COMM_WORLD,1);
    }   

    // Read dataset (only by rank 0)
    start_timer(&timers.io_start);
    if(rank == 0){
        if(read_dataset(inputParams.dataset_filename, &metadata, X, ground_truth_labels) != 0){
            fprintf(stderr, "Failed to read dataset from file: %s\n", inputParams.dataset_filename);
            safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&cluster_acc,&local_gamma);
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
    int local_acc_alloc_status =  alloc_accumulators(&cluster_acc, &metadata); 

    // Check that all allocations were successful
    if (!local_X || !local_gamma || !local_predicted_labels || local_acc_alloc_status != 0) {
        fprintf(stderr, "Local variable memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&cluster_acc, &local_gamma);
        free_accumulators(&local_cluster_acc);
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
        m_step_parallelized(local_X, local_N, &metadata, local_gamma, &cluster_params, &cluster_acc, &local_cluster_acc, rank);
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
        if (inputParams.output_filename){
            int write_status = write_labels_info(inputParams.output_filename, predicted_labels, ground_truth_labels, metadata.N);
            if(write_status != 0){
                fprintf(stderr, "Failed to write labels to file: %s\n", inputParams.output_filename);
            }
        }
    }
    stop_timer(&timers.io_start, &timers.io_time);
    
    // Stop total execution timer
    stop_timer(&timers.start_time, &timers.total_time);
    // Report the execution info (only by rank 0)
    if(rank == 0){
        if(write_execution_info(inputParams.benchmarks_filename, size, &metadata, &timers) != 0){
            fprintf(stderr, "Failed to write execution info to file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Free all the allocated memory
    safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&cluster_params,&cluster_acc,&local_gamma);
    free_accumulators(&local_cluster_acc);
    // Finalize MPI communicator
    MPI_Finalize();
    return 0;
}