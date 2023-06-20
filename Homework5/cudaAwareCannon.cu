/*
 * Damian Franco
 * CS-542
 * Homework 5
 *
 * This program implments the Cuda-Aware 
 * GPU-Accelerated Cannon’s Algorithm
 * https://jrtechs.net/data-science/cuda-vs-cpu-performance 
 */
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 32

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB) {
    // calculate the row and column of the result matrix element
    // that is computed by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= numRowsA || col >= numColsB)
        return; // exit if the thread computes an element outside of the result matrix

    // initialize the result matrix element to 0
    float result = 0;

    // perform the matrix multiplication for this result matrix element
    for (int i = 0; i < numColsA; i++)
    {
        result += A[row * numColsA + i] * B[i * numColsB + col];
    }

    // store the result in the result matrix
    C[row * numColsB + col] = result;
}

void matrixMultiply(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB) {
    // calculate the size of the matrices in bytes
    size_t sizeA = numRowsA * numColsA * sizeof(float);
    size_t sizeB = numColsA * numColsB * sizeof(float);
    size_t sizeC = numRowsA * numColsB * sizeof(float);

    // allocate memory on the GPU for the matrices
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**) &d_A, sizeA);
    cudaMalloc((void**) &d_B, sizeB);
    cudaMalloc((void**) &d_C, sizeC);

    // copy the matrices from host memory to the GPU
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // calculate the dimensions of the grid and blocks for the kernel
    dim3 gridDim((numColsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (numRowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // launch the kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, numRowsA, numColsA, numColsB);

    // copy the result matrix from the GPU to host memory
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // free the memory allocated on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // set up for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // size of matrix
    int n = 1024;

    // initialize the matrices
    float A[n][n] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    float B[n][n] = {{13, 14}, {15, 16}, {17, 18}};
    float C[n][n];

    // memory size of whole matrix
    int size = n*n*sizeof(float);
    
    // allocate matrix memory
    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    // set variables for matrices
    for (int i = 0; i < n*n; i++) {
        A[i] = 0.5;
        B[i] = 0.2;
    }   

    // perform the matrix multiplication
    cudaEventRecord(start);
    matrixMultiply((float*) A, (float*) B, (float*) C, 4, 3, 2);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Size %d, with runtime %8f\n" n, (milliseconds*1000));
    return 0;
}