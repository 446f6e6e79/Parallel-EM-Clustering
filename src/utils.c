#include <headers/utils.h>

static inline void free_and_null(void **p) {
    if (p && *p) {
        free(*p);
        *p = NULL;
    }
}

/*
    Safely frees all allocated memory (passed by address)
*/
void safe_cleanup(
    double **X,
    int **predicted_labels,
    int **ground_truth_labels,
    double **mu,
    double **sigma,
    double **pi,
    double **resp
)
{
    free_and_null((void**)X);
    free_and_null((void**)predicted_labels);
    free_and_null((void**)ground_truth_labels);
    free_and_null((void**)mu);
    free_and_null((void**)sigma);
    free_and_null((void**)pi);
    free_and_null((void**)resp);
}
