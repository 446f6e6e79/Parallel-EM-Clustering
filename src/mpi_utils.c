#include "headers/mpi_utils.h"


/*
    Computes the number of rows assigned to the local process
    Parameters:
*/
int compute_local_N(int N, int size, int rank) {
    int base = N / size;
    return base + (rank < N % size);
}

/*
    Computes counts and displacements for MPI_Scatterv or MPI_Gatherv.
    factor: multiplier for each element (e.g., D for scatter, 1 for labels)
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


*/
void scatter_dataset(double *X, double *local_X, int N, int local_N, int D, int rank, int size) {
    int *counts = NULL;             // Number of elements to send to each process. sendcounts[i] = number of elements sent to process i
    int *displs = NULL;             // Displacements for each process. displs[i] = offset in the send buffer from which to take the elements for process i
    
    if(rank == 0){
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if(!counts || !displs){
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    if (rank == 0) {
        compute_counts_displs(N, size, D, counts, displs);
    }
    
    MPI_Scatterv(X, counts, displs, MPI_DOUBLE,
                 local_X, local_N * D, MPI_DOUBLE, 0, MPI_COMM_WORLD); // local count is ignored by MPI, must match allocation

    if(rank == 0){
        free(counts);
        free(displs);
        counts = NULL;
        displs = NULL;
    }
}

void gather_dataset(int *local_predicted_labels, int *predicted_labels, int N, int local_N, int rank, int size) {
    int *counts = NULL;             // Number of elements to send to each process. sendcounts[i] = number of elements sent to process i
    int *displs = NULL;             // Displacements for each process. displs[i] = offset in the send buffer from which to take the elements for process i
    
    if(rank == 0){
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        if(!counts || !displs){
                fprintf(stderr, "Memory allocation failed\n");
                MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    if (rank == 0) {
        compute_counts_displs(N, size, 1, counts, displs);
    }
    
    MPI_Gatherv(local_predicted_labels, local_N, MPI_INT,
                predicted_labels, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

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
 *     - N: Pointer to number of samples (input on root, output on all processes)
 *     - D: Pointer to number of features (input on root, output on all processes)  
 *     - K: Pointer to number of clusters (input on root, output on all processes)
 */
void broadcast_metadata(int *N, int *D, int *K, int rank) {
    int metadata[3];
    // Root process packs the metadata
    if (rank == 0) {
        metadata[0] = *N;
        metadata[1] = *D;
        metadata[2] = *K;
    }
    
    // Single MPI call to broadcast all metadata
    MPI_Bcast(metadata, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    // All processes unpack the metadata
    if(rank != 0){
        *N = metadata[0];
        *D = metadata[1];
        *K = metadata[2];
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
 *     - K: Number of clusters
 *     - D: Number of features
 */
void broadcast_clusters_parameters(double *mu, double *sigma, double *pi, int K, int D) {
    MPI_Datatype mpi_param_type;
    int blocklengths[3] = {K * D, K * D, K};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Aint base_address;
    MPI_Get_address(mu, &base_address);
    MPI_Get_address(sigma, &displacements[1]);
    MPI_Get_address(pi, &displacements[2]);

    displacements[0] = 0;
    displacements[1] -= base_address;
    displacements[2] -= base_address;

    MPI_Type_create_struct(3, blocklengths, displacements, types, &mpi_param_type);
    MPI_Type_commit(&mpi_param_type);

    // Broadcast from process 0
    MPI_Bcast(mu, 1, mpi_param_type, 0, MPI_COMM_WORLD);

    // Free the derived datatype
    MPI_Type_free(&mpi_param_type);
}