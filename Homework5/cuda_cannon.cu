/*
 * Damian Franco
 * CS-542
 * Homework 5
 *
 * This program implments the Copy to CPU and
 * Cuda-Aware GPU-Accelerated Cannon’s Algorithm.
 * https://jrtechs.net/data-science/cuda-vs-cpu-performance 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#include "mpi_cannon.hpp"

void copy_to_cpu_cannon(float* A, float* B, float* C,
        int n, int sq_num_procs, int rank_row, int rank_col)
{
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




void cuda_aware_cannon(float* A, float* B, float* C,
        int n, int sq_num_procs, int rank_row, int rank_col)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int size = n*n;

    float* send_A = new float[size];
    float* recv_A = new float[size];
    float* send_B = new float[size];
    float* recv_B = new float[size];

    int send_proc_A, send_proc_B;
    int recv_proc_A, recv_proc_B;
    int tag_a = 1234;
    int tag_b = 4321;

    memset(C, 0, size*sizeof(float));

    // Initial Shift : 
    get_init_procs(rank_row, rank_col, sq_num_procs,
            &send_proc_A, &send_proc_B, &recv_proc_A, &recv_proc_B);
    communicate(send_proc_A, recv_proc_A, tag_a, size, 
            rank_row && rank_col / rank_row % 2 == 0, A, recv_A);
    communicate(send_proc_B, recv_proc_B, tag_b, size, 
            rank_col && rank_row / rank_col % 2 == 0, B, recv_B);
    matmat(n, recv_A, recv_B, C);

    // Send and recv A and B from neighborhing processes in proc grid
    get_rotation_procs(rank_row, rank_col, sq_num_procs,
            &send_proc_A, &send_proc_B, &recv_proc_A, &recv_proc_B);
    for (int i = 1; i < sq_num_procs; i++)
    {
        swap(&send_A, &recv_A, &send_B, &recv_B);
        communicate(send_proc_A, recv_proc_A, tag_a, size, rank_col % 2 == 0,
                send_A, recv_A);
        communicate(send_proc_B, recv_proc_B, tag_b, size, rank_row % 2 == 0,
                send_B, recv_B);
        matmat(n, recv_A, recv_B, C);
    }

    delete[] send_A;
    delete[] recv_A;
    delete[] send_B;
    delete[] recv_B;
}