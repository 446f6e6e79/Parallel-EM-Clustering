#ifndef FILE_IO_H
#define FILE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/file.h>
#include <string.h>

#define READING_BUFFER_SIZE 2048

int read_metadata(const char *metadata_filename, int *samples, int *features, int *clusters);
int read_dataset(const char *filename, double *examples_buffer, int *labels_buffer, int num_features, int num_samples);
int write_execution_info(const char *filename, int n_process, int n_elements, double time_seconds, double io_time, double compute_time);

#endif
