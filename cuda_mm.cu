#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matMul(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
            sum += A[row*n + k] * B[k*n + col];
        C[row*n + col] = sum;
    }
}

void run_case(int n) {
    int size = n*n;
    size_t bytes = size * sizeof(double);

    double *hA = (double*)malloc(bytes);
    double *hB = (double*)malloc(bytes);
    double *hC = (double*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        hA[i] = 1.0;
        hB[i] = 1.0;
    }

    double *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMul<<<blocks, threads>>>(dA, dB, dC, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("CUDA  | Size=%4d | Time=%8.4f ms\n", n, ms);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
}

int main() {
    int sizes[] = {256, 512, 1024};

    printf("=== CUDA Results (GPU) ===\n");
    for (int i = 0; i < 3; i++)
        run_case(sizes[i]);

    return 0;
}
