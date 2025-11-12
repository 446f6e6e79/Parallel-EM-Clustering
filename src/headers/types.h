#ifndef TYPES_H
#define TYPES_H

typedef struct
{
    // Total execution time of the program
    double start_time;
    double total_time;

    // I/O operation time
    double io_start;
    double io_time;

    // Computation time
    double compute_start;
    double compute_time;

    // Initial data distribution time
    double data_distribution_start;
    double data_distribution_time;

    // E-step time
    double e_step_start;
    double e_step_time;

    // M-step time
    double m_step_start;
    double m_step_time;
} Timers_t;

typedef struct {
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    int max_line_size;                  // Maximum number of character in a line in the dataset file
} Metadata;

#endif