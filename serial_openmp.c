#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void run_serial(int n) {
    int size = n * n;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *C = malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
    }

    double end = omp_get_wtime();
    printf("Serial  | Size=%4d | Time=%8.6f sec\n", n, end - start);

    free(A); free(B); free(C);
}

void run_openmp(int n, int threads) {
    int size = n * n;
    double *A = malloc(size * sizeof(double));
    double *B = malloc(size * sizeof(double));
    double *C = malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
    }

    double end = omp_get_wtime();
    printf("OpenMP | Size=%4d | Threads=%2d | Time=%8.6f sec\n",
           n, threads, end - start);

    free(A); free(B); free(C);
}

int main() {
    int sizes[] = {256, 512, 1024};
    int threads[] = {2, 4, 8};

    printf("=== CPU Results (Serial + OpenMP) ===\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        run_serial(n);
        for (int t = 0; t < 3; t++)
            run_openmp(n, threads[t]);
        printf("-------------------------------------\n");
    }
    return 0;
}
