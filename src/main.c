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
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    int max_line_size;                  // Maximum number of character in a line in the dataset file
    
    double *X = NULL;                   // X[N * D] Vector of data points
    double *mu = NULL;                  // mu[k * D] Vector of feature means per cluster; mu[k * D + d] is the mean of feature d for cluster k
    double *sigma = NULL;               // sigma[k * D] Vector of variance per each feature per cluster; sigma[k * D + d] is the variance of feature d for cluster k
    double *pi = NULL;                  // pi[k] Vector of mixture weights: prior probability that a random data point belongs to cluster k
    
    double *local_gamma = NULL;         // local_gamma[local_N * K] Local responsibilities vector for each process. local_gamma[i * K + k] is the responsibility of cluster k for data point i
    double *N_k = NULL;                 // N_k[k] = Sum of responsibilities per cluster
    double *mu_k = NULL;                // mu_k[k * D] = Weighted sums for means
    double *sigma_k = NULL;             // sigma_k[k * D] = Weighted sums for variances

    int *predicted_labels = NULL;       // Predicted cluster labels
    int *local_predicted_labels = NULL; // Local predicted cluster labels
    int *ground_truth_labels = NULL;    // Ground truth labels

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
    
    // Allocate buffers
    int alloc_fail = 0;
    if(rank == 0){
        X = malloc(N * D * sizeof(double));
        predicted_labels = malloc(N * sizeof(int));
        ground_truth_labels = malloc(N * sizeof(int));
        if(!X || !predicted_labels || !ground_truth_labels) {
            alloc_fail = 1;
        }
    }
    //TODO: check that all these malloc are needed by all processes or just by process zero
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
    
    printf("Process %d\n", rank);

    //TODO: only rank 0 initializes the parameters, then broadcast to all processes
    //Initialize parameters for the EM algorithm
    if (rank == 0) init_params(X, N, D, K, mu, sigma, pi);

    // Prepare for scattering data
    data_distribution_start = MPI_Wtime();
    //TODO: once verified that data distribution time is correctly measured, refactor to a function
    int *sendcounts = NULL;         // Number of elements to send to each process. sendcounts[i] = number of elements sent to process i
    int *displs = NULL;             // Displacements for each process. displs[i] = offset in the send buffer from which to take the elements for process i
    //TODO: check all the sendcounts, displs and local_n
    // Root process prepares sendcounts and displs arrays
    if (rank == 0) {
        // Allocate the two vectors
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int rem = N % size;
        int offset = 0;
        // Calculate sendcounts and displs
        for (int i = 0; i < size; i++) {
            // Calculate the number of local samples for each process
            int n_local = N / size;
            // Distribute the remainder among the first 'rem' processes
            if (i < rem) n_local++;
            // Each process gets n_local * D elements
            sendcounts[i] = n_local * D;
            // Displacement is the cumulative sum of previous sendcounts
            displs[i] = offset;
            offset += n_local * D;
        }
    }
    // Compute the number of local samples for each process
    int local_N = N / size;
    // Compute the remainder and distribute it
    if (rank < N % size) local_N++;

    //Allocate the vectors for local data
    double *local_X = malloc(local_N * D * sizeof(double));
    local_gamma = malloc(local_N * K * sizeof(double));
    local_predicted_labels = malloc(local_N * sizeof(int)); 
    if (!local_X || !local_gamma || !local_predicted_labels) {
        fprintf(stderr, "Memory allocation failed\n");
        safe_cleanup(&X,&predicted_labels,&ground_truth_labels,&mu,&sigma,&pi,&local_gamma,&N_k,&mu_k,&sigma_k);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    // Scatter the data points to all processes
    MPI_Scatterv(
        X, sendcounts, displs, MPI_DOUBLE,
        local_X, local_N * D, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    
    // Free sendcounts and displs as they are no longer needed
    if (rank == 0) {
        free(sendcounts);
        sendcounts = NULL;
        free(displs);
        displs = NULL;
    }
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

        //TODO: check how to parallelize
        //M-step
        double m_step_start = MPI_Wtime();
        m_step(X, N, D, K, local_gamma,  mu, sigma, pi, N_k, mu_k, sigma_k);
        m_step_time += MPI_Wtime() - m_step_start;
    }
    
    // Compute local predicted labels from responsibilities
    compute_clustering(local_gamma, local_N, K, local_predicted_labels);
    compute_time += MPI_Wtime() - compute_start;

    int *receive_counts = NULL;
    displs = NULL;
    if(rank == 0){
        receive_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int rem = N % size;
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int n_local = N / size;
            if (i < rem) n_local++;
            receive_counts[i] = n_local;
            displs[i] = offset;
            offset += n_local;
        }
    }
    // Gather predicted labels from all processes
    MPI_Gatherv(local_predicted_labels, local_N, MPI_INT,
                predicted_labels, receive_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

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
    return 0;
}