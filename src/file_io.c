#include "headers/file_io.h"

/*
    Read the dataset from the given filename, storing information into the provided vectors.
    The dataset file is expected to have a header line followed by num_samples data lines.
    Each data line contains num_features feature values followed by a label, all separated by commas.

    Parameters:
        - filename: Path to the dataset file.
        - num_features: Number of features per sample.
        - num_samples: Number of samples to read.
        - max_line_size: Maximum number of characters in a line.
    Output parameters:
        - examples_buffer: Array to store feature values.
        - labels_buffer: Array to store labels.

    Returns:
        0 on success, -1 on failure.
*/
int read_dataset(const char *filename, int num_features, int num_samples,  int max_line_size, double *examples_buffer, int *labels_buffer) {
    // Open the dataset file
    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "Error opening dataset file \n");
        return -1;
    }
    // Size of the allocated reading buffer. We should take into account the \n and \0 characters
    int BUFFER_SIZE = max_line_size + 3;
    // Creating reading buffer. We have to take into account the \n and \0 characters
    char *line_buffer = malloc(BUFFER_SIZE * sizeof(char));
    int readed_rows = 0;

    // Skip the first line (header)
    if(!fgets(line_buffer, BUFFER_SIZE, fp)) return -1;

    // Read each line and parse the values
    while(readed_rows < num_samples && fgets(line_buffer, BUFFER_SIZE, fp) != NULL){
        // Parse the line, extracting features and label
        char *ptr = line_buffer;
        // For each feature, add to the examples buffer
        for(int f = 0; f < num_features; f++){
            examples_buffer[readed_rows * num_features + f] = strtod(ptr,&ptr);
            if(*ptr == ',') ptr++;
        }
        // Add the ground truth labels
        labels_buffer[readed_rows] = (int)strtol(ptr, NULL, 10);
        readed_rows++;
    }
    // Close the filePointer
    fclose(fp);

    // Return 0 if successfully read
    return readed_rows == num_samples ? 0 : -1;
}

/*
    Read metadata from the given metadata file.
    The metadata file is expected to contain lines in the format:
        samples: <number_of_samples>
        features: <number_of_features>
        clusters: <number_of_clusters>

    Parameters:
        - metadata_filename: Path to the metadata file.
    Output parameters:
        - samples*: Pointer to store the number of samples.
        - features*: Pointer to store the number of features.
        - clusters*: Pointer to store the number of clusters.
        - max_line_size*: Pointer to store the maximum line size.
    Returns:
        0 on success, -1 on failure.
*/
int read_metadata(const char *metadata_filename, int *samples, int *features, int *clusters, int *max_line_size) {
    // Open the metadata file
    FILE *meta_fp = fopen(metadata_filename, "r");
    if(!meta_fp){
        fprintf(stderr, "Error opening metadata file \n");
        return -1;
    }

    // Read metadata lines
    char line[READING_BUFFER_SIZE];
    while(fgets(line, sizeof(line), meta_fp) != NULL) {
        if(sscanf(line, "samples: %d", samples) == 1) continue;
        if(sscanf(line, "features: %d", features) == 1) continue;
        if(sscanf(line, "clusters: %d", clusters) == 1) continue;
        if(sscanf(line, "max_line_size: %d", max_line_size) == 1) continue;
    }

    fclose(meta_fp);
    // Check that all metadata values were correctly read and valid
    if(*samples <= 0 || *features <= 0 || *clusters <= 0 || *max_line_size<=0) return -1;
    return 0;
}

/*
    Write the execution information to the output csv file
    The file format is:
        n_process,total_size,time_seconds
    Each execution will append a new line to the file.
    Returns 0 on success, -1 on failure.
*/
int write_execution_info(const char *filename, int n_process, int n_elements, double time_seconds, double io_time, double compute_time) {
    // Open the file in append mode
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "File size is not a multiple of IP address size\n");
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

    // Write the execution info (note: MPI_Offset is typically a long long)
    if (fprintf(fp, "%d,%d,%.6f,,%.6f,,%.6f\n", n_process, n_elements, time_seconds, io_time, compute_time) == -1) {
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
    Write predicted and real labels to a CSV file for validation.
    The CSV file will have two columns: "predicted" and "real".

    Parameters:
        - filename: Path to the output CSV file.
        - predicted_labels: Array of predicted cluster labels.
        - real_labels: Array of real labels.
        - num_samples: Number of samples (length of the label arrays).

    Returns:
        0 on success, -1 on failure.
*/
int write_labels_info(const char *filename, int *predicted_labels, int *real_labels, int num_samples){
    // Open the file for writing
    FILE *f = fopen(filename, "w");
    if(!f){
        perror("fopen");
        return -1;
    }
    // Print CSV header
    fprintf(f, "predicted,real\n");

    // Write each label pair
    for(int i=0; i<num_samples; i++){
        fprintf(f, "%d,%d\n", predicted_labels[i], real_labels[i]);
    }
    fclose(f);
    return 0;
}
