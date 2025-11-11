#ifndef FILE_IO_H
#define FILE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include "utils.h"

#define READING_BUFFER_SIZE 256

int read_metadata(const char *metadata_filename, int *samples, int *features, int *clusters, int *max_line_size);
int read_dataset(const char *filename, int num_features, int num_samples,  int max_line_size, double *examples_buffer, int *labels_buffer);
int write_execution_info(const char *filename, int n_process, int n_samples, int n_features, int n_clusters, Timers_t *timers);
int write_labels_info(const char *filename, int *predicted_labels, int *real_labels, int num_samples);

#endif
