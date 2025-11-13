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

int read_metadata(const char *meta_data_file_path, Metadata *metadata);
int read_dataset(const char *filename, Metadata *metadata, double *examples_buffer, int *labels_buffer);
int write_execution_info(const char *filename, int n_process, Metadata *metadata, Timers_t *timers);
int write_labels_info(const char *filename, double *X, int *predicted_labels, int *real_labels, Metadata *metadata, ClusterParams *cluster_params, int iteration, char mode);
#endif
