#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "types.h"

#define READING_BUFFER_SIZE 256

int read_metadata(const char *metadata_filename, Metadata *metadata);
int read_dataset(const char *filename, Metadata *metadata, double *examples_buffer, int *labels_buffer);
int write_execution_info(const char *filename, int n_process, Metadata *metadata, Timers_t *timers);
int write_labels_info(const char *filename, int *predicted_labels, int *real_labels, int num_samples);
void debug_print_first_samples(Metadata *metadata, double *X, int *ground_truth_labels);
void debug_print_scatter(int local_N, int D, double *local_X, int rank);
void debug_print_cluster_params(Metadata *metadata, ClusterParams *cluster_params);

#endif
