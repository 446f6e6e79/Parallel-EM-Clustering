#ifndef TYPES_H
#define TYPES_H

/*
    Metadata structure to hold dataset parameters
*/
typedef struct {
    int N;                              // Number of samples
    int D;                              // Number of features
    int K;                              // Number of clusters
    int max_line_size;                  // Maximum number of character in a line in the dataset file
} Metadata;

/*
    Cluster parameters structure
*/
typedef struct {
    double *mu;                         // mu[k * D] = Mean of cluster k for each feature
    double *sigma;                      // sigma[k * D] = Variance of cluster k for each feature
    double *pi;                         // pi[k] = Mixture weight of cluster k
} ClusterParams;

/*
    Cluster parameters accumulators structure
*/
typedef struct {
    double *N_k;                        // N_k[k] = Sum of responsibilities per cluster
    double *mu_k;                       // mu_k[k * D] = Weighted sums for means
    double *sigma_k;                    // sigma_k[k * D] = Weighted sums for variances
} Accumulators;

/*
    Timers structure to keep track of different execution times
*/
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

/*

*/
typedef struct {
    // Path to the input file containing the dataset
    const char *dataset_filename;
    // Path to the metadata file
    const char *metadata_filename;
    // Path to the output file containing execution information (optional)
    const char *benchmarks_filename;
    // Path to the output file for predicted labels (optional)
    const char *output_filename;
} InputParams_t;

#endif