#ifndef FILE_IO_H
#define FILE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/file.h>

#include <string.h>

int write_execution_info(const char *filename, int n_process, int n_elements, double time_seconds, double io_time, double compute_time);

#endif
