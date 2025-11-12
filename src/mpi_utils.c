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
void scatter_dataset(double *X, double *local_X, int local_N, Metadata *metadata, int rank, int size) {
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
        compute_counts_displs(metadata->N, size, metadata->D, counts, displs);
    }
    
    MPI_Scatterv(X, counts, displs, MPI_DOUBLE,
                 local_X, local_N * metadata->D, MPI_DOUBLE, 0, MPI_COMM_WORLD); // local count is ignored by MPI, must match allocation

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
 *     - metadata:
 *     - rank:
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
 *     - metadata:
 *       - K: Number of clusters
 *       - D: Number of features
 */
void broadcast_clusters_parameters(double *mu, double *sigma, double *pi, Metadata *metadata) {
    MPI_Datatype mpi_param_type;
    int blocklengths[3] = {metadata->K * metadata->D, metadata->K * metadata->D, metadata->K};
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