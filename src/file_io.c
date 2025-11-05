#include "headers/file_io.h"

/*
    Write the execution information to the output csv file
    The file format is:
        n_process,total_size,time_seconds
    Each execution will append a new line to the file.
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
