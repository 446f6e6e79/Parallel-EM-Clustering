#include "headers/mpi_utils.h"


/*
    Computes the number of rows assigned to the local process
    Parameters:
        - N: Total number of samples
        - size: Total number of MPI processes
        - rank: Rank of the current MPI process
*/
int compute_local_N(int N, int size, int rank) {
    int base = N / size;
    return base + (rank < N % size);
}

/*
    Computes counts and displacements for MPI_Scatterv or MPI_Gatherv.
    Parameters:
        - N: Total number of samples
        - size: Total number of MPI processes
        - factor: Multiplicative factor for each count (e.g., D for data points, 1 for labels)
        - counts: Output array of size 'size' to hold counts for each process
        - displs: Output array of size 'size' to hold displacements for each process
*/
void compute_counts_displs(int N, int size, int factor, int *counts, int *displs) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int n_local = compute_local_N(N, size, i);
        counts[i] = n_local * factor;
        displs[i] = offset;
        offset += n_local * factor;
    }
}

/*
    Scatter dataset X from root process to all other processes.
    Each process receives local_N rows in local_X.
    
    Parameters:
     - X: (N x D) Dataset matrix (only valid on root process)
     - local_X: (local_N x D) Local dataset matrix for each process
     - local_N: Number of rows assigned to the local process
     - metadata: structure containing N, D, K
     - rank: Rank of the current MPI process
     - size: Total number of MPI processes
*/
void scatter_dataset(double *X, double *local_X, int local_N, Metadata *metadata, int rank, int size) {
    // Number of elements to send to each process. sendcounts[i] = number of elements sent to process i
    int *counts = NULL;          
     // Displacements for each process. displs[i] = offset in the send buffer from which to take the elements for process i   
    int *displs = NULL;            
    // Allocate counts and displs only on root process
    if(rank == 0){
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if(!counts || !displs){
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    // Compute counts and displs on root process
    if (rank == 0) {
        compute_counts_displs(metadata->N, size, metadata->D, counts, displs);
    }
    // Scatter the dataset
    MPI_Scatterv(X, counts, displs, MPI_DOUBLE,
                 local_X, local_N * metadata->D, MPI_DOUBLE, 0, MPI_COMM_WORLD); // local count is ignored by MPI, must match allocation
    // Free counts and displs on root process
    if(rank == 0){
        free(counts);
        free(displs);
        counts = NULL;
        displs = NULL;
    }
}

/*
    Gather predicted labels from all processes to root process.
    
    Parameters:
     - local_predicted_labels: (local_N) Local predicted labels for each process
     - predicted_labels: (N) Global predicted labels (only valid on root process)
     - N: Total number of samples
     - local_N: Number of rows assigned to the local process
     - rank: Rank of the current MPI process
     - size: Total number of MPI processes
*/
void gather_dataset(int *local_predicted_labels, int *predicted_labels, int N, int local_N, int rank, int size) {
    // Number of elements to send to each process. sendcounts[i] = number of elements sent to process i
    int *counts = NULL;   
     // Displacements for each process. displs[i] = offset in the send buffer from which to take the elements for process i          
    int *displs = NULL;            
    // Allocate counts and displs only on root process
    if(rank == 0){
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if(!counts || !displs){
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    // Compute counts and displs on root process
    if (rank == 0) {
        compute_counts_displs(N, size, 1, counts, displs);
    }
    // Gather the predicted labels
    MPI_Gatherv(local_predicted_labels, local_N, MPI_INT,
                predicted_labels, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    // Free counts and displs on root process
    if(rank == 0){
        free(counts);
        free(displs);
        counts = NULL;
        displs = NULL;
    }
}


/**
 *   Broadcast metadata (N, D, K) from root process to all other processes
 *    using a single MPI_Bcast operation for efficiency.
 *    
*    Parameters:    
*    - metadata: Metadata structure containing N, D, and K
*    - rank: Rank of the current MPI process
 */
void broadcast_metadata(Metadata *metadata, int rank) {
    int meta_array[3];
    // Root process packs the metadata
    if (rank == 0) {
        meta_array[0] = metadata->N;
        meta_array[1] = metadata->D;
        meta_array[2] = metadata->K;
    }
    // Broadcast all metadata in one MPI call
    MPI_Bcast(meta_array, 3, MPI_INT, 0, MPI_COMM_WORLD);
    // All other processes unpack the metadata
    if (rank != 0) {
        metadata->N = meta_array[0];
        metadata->D = meta_array[1];
        metadata->K = meta_array[2];
    }
}

/**
 *   Broadcast model parameters (mu, sigma, pi) from process 0 to all other processes
 *    using an MPI derived datatype to send all parameters in a single operation.
 *    
 *    Parameters:
 *     - mu: (K x D) Matrix of cluster means
 *     - sigma: (K x D) Matrix of cluster variances  
 *     - pi: (K) Vector of mixture weights
 *     - metadata: Metadata structure containing N, D, and K
 */
void broadcast_clusters_parameters(double *mu, double *sigma, double *pi, Metadata *metadata) {
    // Create MPI derived datatype for cluster parameters
    MPI_Datatype mpi_param_type;
    // Define block lengths, displacements, and types
    int blocklengths[3] = {metadata->K * metadata->D, metadata->K * metadata->D, metadata->K};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    // Calculate displacements
    MPI_Aint base_address;
    MPI_Get_address(mu, &base_address);
    MPI_Get_address(sigma, &displacements[1]);
    MPI_Get_address(pi, &displacements[2]);
    // Adjust displacements relative to base address
    displacements[0] = 0;
    displacements[1] -= base_address;
    displacements[2] -= base_address;
    // Create and commit the derived datatype
    MPI_Type_create_struct(3, blocklengths, displacements, types, &mpi_param_type);
    MPI_Type_commit(&mpi_param_type);

    // Broadcast from process 0
    MPI_Bcast(mu, 1, mpi_param_type, 0, MPI_COMM_WORLD);

    // Free the derived datatype
    MPI_Type_free(&mpi_param_type);
}