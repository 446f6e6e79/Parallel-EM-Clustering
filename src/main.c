#include <stdio.h>
#include <stdlib.h>
#include "headers/file_io.h"

/*
    Expection-Maximization Clustering Algorithm

    Usage: ./program <dataset_file> <metadata_file>
*/
int main(int argc, char **argv) {
    // Check command line arguments
    if(argc < 3){
        fprintf(stderr, "Usage: %s <dataset_file> <metadata_file>\n", argv[0]);
        return 1;
    }
    if(argv[1] == NULL || argv[2] == NULL){
        fprintf(stderr, "Dataset file and metadata file must be provided\n");
        return 1;
    }

    // Get filenames from arguments
    const char *filename = argv[1];
    const char *metadata_filename = argv[2];

    // Read metadata
    int samples = 0, features = 0, clusters = 0;
    int meta_status = read_metadata(metadata_filename, &samples, &features, &clusters);
    if(meta_status != 1){
        fprintf(stderr, "Failed to read metadata from file: %s\n", metadata_filename);
        return 1;
    }
    printf("Metadata: samples=%d, features=%d, clusters=%d\n", samples, features, clusters);

    // Allocate buffers
    double *examples_buffer = malloc(samples * features * sizeof(double));
    int *labels_buffer = malloc(samples * sizeof(int));

    // Read dataset
    int n_read = read_dataset(filename, examples_buffer, labels_buffer, features, samples);
    if(n_read != 1){
        fprintf(stderr, "Failed to read dataset from file: %s\n", filename);
        free(examples_buffer);
        free(labels_buffer);
        return 1;
    }
    printf("dataset read -> %d rows\n", samples);

    // test print first few rows
    for(int i=0; i<5 && i<samples; i++){
        printf("Row %d: ", i);
        for(int f=0; f<features; f++){
            printf("%lf ", examples_buffer[i * features + f]);
        }
        printf("Label=%d\n", labels_buffer[i]);
    }

    // free memory
    free(examples_buffer);
    free(labels_buffer);

    return 0;
}
