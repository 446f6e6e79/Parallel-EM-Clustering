#include "headers/io_utils.h"

/*
    Read the dataset from the given filename, storing information into the provided vectors.
    The dataset file is expected to have a header line followed by num_samples data lines.
    Each data line contains num_features feature values followed by a label, all separated by commas.

    Parameters:
        - filename: Path to the dataset file.
        - metadata contains:
            - num_features: Number of features per sample.
            - num_samples: Number of samples to read.
            - max_line_size: Maximum number of characters in a line.
    Output parameters:
        - examples_buffer: Array to store feature values.
        - labels_buffer: Array to store labels.

    Returns:
        0 on success, -1 on failure.
*/
int read_dataset(const char *filename, Metadata *metadata, double *examples_buffer, int *labels_buffer) {
    // Open the dataset file
    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "Error opening dataset file \n");
        return -1;
    }
    // Size of the allocated reading buffer. We should take into account the \n and \0 characters
    int BUFFER_SIZE = metadata->max_line_size + 3;
    // Creating reading buffer. We have to take into account the \n and \0 characters
    char *line_buffer = malloc(BUFFER_SIZE * sizeof(char));
    int readed_rows = 0;

    // Skip the first line (header)
    if(!fgets(line_buffer, BUFFER_SIZE, fp)) return -1;

    // Read each line and parse the values
    while(readed_rows < metadata->N && fgets(line_buffer, BUFFER_SIZE, fp) != NULL){
        // Parse the line, extracting features and label
        char *ptr = line_buffer;
        // For each feature, add to the examples buffer
        for(int f = 0; f < metadata->D; f++){
            examples_buffer[readed_rows * metadata->D + f] = strtod(ptr,&ptr);
            if(*ptr == ',') ptr++;
        }
        // Add the ground truth labels
        labels_buffer[readed_rows] = (int)strtol(ptr, NULL, 10);
        readed_rows++;
    }
    // Close the filePointer
    fclose(fp);

    // Return 0 if successfully read
    return readed_rows == metadata->N ? 0 : -1;
}

/*
    Read metadata from the given metadata file.
    The metadata file is expected to contain lines in the format:
        samples: <number_of_samples>
        features: <number_of_features>
        clusters: <number_of_clusters>

    Parameters:
        - meta_data_file_path: Path to the metadata file.
    Output parameters:
        - metadata: Struct containing number of samples, number of features, number of clusters and max line size
    Returns:
        0 on success, -1 on failure.
*/
int read_metadata(const char *meta_data_file_path, Metadata *metadata) {
    // Open the metadata file
    FILE *meta_fp = fopen(meta_data_file_path, "r");
    if(!meta_fp){
        fprintf(stderr, "Error opening metadata file \n");
        return -1;
    }

    // Read metadata lines
    char line[READING_BUFFER_SIZE];
    while(fgets(line, sizeof(line), meta_fp) != NULL) {
        if(sscanf(line, "samples: %d", &metadata->N) == 1) continue;
        if(sscanf(line, "features: %d", &metadata->D) == 1) continue;
        if(sscanf(line, "clusters: %d", &metadata->K) == 1) continue;
        if(sscanf(line, "max_line_size: %d", &metadata->max_line_size) == 1) continue;
    }

    fclose(meta_fp);
    // Check that all metadata values were correctly read and valid
    if(metadata->N <= 0 || metadata->D <= 0 || metadata->K <= 0 || metadata->max_line_size<=0) return -1;
    return 0;
}

/*
    Write the execution information to the output csv file
    The file format is:
        n_process,n_samples,n_features,n_clusters,time_seconds,io_time,compute_time,data_distribution_time,e_step_time,m_step_time,data_distribution_time
    Each execution will append a new line to the file.
    Parameters:
        - filename: Path to the output csv file.
        - n_process: Number of processes used in the execution.
        - metadata: Metadata structure containing dataset information.
        - timers: Timers structure containing execution times.
    Returns:
        0 on success, -1 on failure.
*/
int write_execution_info(const char *filename, int n_process, Metadata *metadata, Timers_t *timers){
    // Open the file in append mode
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "Error opening execution info file\n");
        return -1;
    }

    // Getting the POSIX file descriptor from the FILE pointer
    int fd = fileno(fp);
    if (fd == -1) {
        fprintf(stderr, "Failed to get file descriptor\n");
        fclose(fp);
        return -1;
    }
    
    // Lock the file for exclusive access
    if (flock(fd, LOCK_EX) == -1) {
        fprintf(stderr, "Failed to lock file\n");
        fclose(fp);
        return -1;
    }

    // If the file is empty, write the header
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size == 0) {
        if (fprintf(fp, "n_process,n_samples,n_features,n_clusters,time_seconds,io_time,compute_time,e_step_time,m_step_time,data_distribution_time\n") == -1) {
            fprintf(stderr, "Failed to write header to file\n");
            flock(fd, LOCK_UN);
            fclose(fp);
            return -1;
        }
    }
    // Write the execution info (note: MPI_Offset is typically a long long)
    if (fprintf(fp, "%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n_process, metadata->N, metadata->D, metadata->K, timers->total_time, timers->io_time, timers->compute_time, timers->e_step_time, timers->m_step_time, timers->data_distribution_time) == -1) {
        fprintf(stderr, "Failed to write to file\n");
        flock(fd, LOCK_UN);
        fclose(fp);
        return -1;
    }

    // Flush to ensure the data is actually written
    fflush(fp);
    
    // Release lock and close the file
    flock(fd, LOCK_UN);       
    fclose(fp);
    return 0;
}

/*
    Write predicted and real labels to a CSV file for validation, in the the specified mode.
    The CSV file will have the following format:
        feature_1, ..., feature_D, real_cluster, predicted_cluster, mu_1, ..., mu_D, sigma_1, ..., sigma_D, pi, iteration

    Parameters:
        - filename: Path to the output CSV file.
        - predicted_labels: Array of predicted cluster labels.
        - real_labels: Array of real labels.
        - metadata: Metadata structure containing dataset information.
        - cluster_params: Cluster parameters structure containing cluster information.
        - iteration: Current iteration number.
        - mode: 'w' to write (overwrite) or 'a' to append to the file.
    Returns:
        0 on success, -1 on failure.
*/
int write_labels_info(const char *filename, double *X, int *predicted_labels, int *real_labels, Metadata *metadata, ClusterParams *cluster_params, int iteration, char mode) {
    // Open the file in the specified mode
    FILE *f = fopen(filename, mode == 'a' ? "a" : "w");
    if(!f){
        perror("fopen");
        return -1;
    }
    // Get the size of the file
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    
    // If the file is empty, write the header
    if (file_size == 0) {
        for(int d = 0; d < metadata->D; d++){
            fprintf(f, "feature_%d,", d+1);
        }
        fprintf(f, "real_cluster,predicted_cluster,");
        for(int d = 0; d < metadata->D; d++){
            fprintf(f, "mu_k_%d,sigma_k_%d,", d+1, d+1);
        }
        fprintf(f, "pi_k,iteration\n");
    }

    // For each sample, write features, predicted label, real label, and cluster parameters
    for(int i = 0; i < metadata->N; i++){
        // Write features
        for(int d = 0; d < metadata->D; d++){
            fprintf(f, "%f,", X[i * metadata->D + d]);
        }
        // Write predicted and real labels
        fprintf(f, "%d,%d,", real_labels[i], predicted_labels[i]);
        // Write cluster parameters of the predicted cluster
        int k = predicted_labels[i];
        for(int d = 0; d < metadata->D; d++){
            fprintf(f, "%f,%f,", cluster_params->mu[k * metadata->D + d], cluster_params->sigma[k * metadata->D + d]);
        }
        fprintf(f, "%f,%d\n", cluster_params->pi[k], iteration);
    }
    fclose(f);
    return 0;
}