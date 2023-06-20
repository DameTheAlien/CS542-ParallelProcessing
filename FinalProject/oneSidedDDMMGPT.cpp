#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

using namespace std;

const int N = 4;

void fillMatrix(double **matrix)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = rand() % 100;
}

void printMatrix(double **matrix)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2)
    {
        if (rank == 0)
            cout << "This program must be run with exactly two processes!" << endl;
        MPI_Finalize();
        return 0;
    }

    double **A = new double*[N];
    double **B = new double*[N];
    double **C = new double*[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = new double[N];
        B[i] = new double[N];
        C[i] = new double[N];
    }

    // Fill matrices A and B with random values
    if (rank == 0)
    {
        fillMatrix(A);
        fillMatrix(B);
    }

    // Create a window object for one-sided communication
    MPI_Win win;
    MPI_Win_create(B[0], N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Compute the matrix-matrix multiplication
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i][j] = 0;
                for (int k = 0; k < N; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }

        // Share the result matrix with the other process
        MPI_Win_fence(0, win);
        for (int i = 0; i < N; i++)
            MPI_Put(C[i], N, MPI_DOUBLE, 1, i * N, N, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
    }
    else
    {
        // Receive the result matrix from the other process
        MPI_Win_fence(0, win);
        for (int i = 0; i < N; i++)
            MPI_Get(C[i], N, MPI_DOUBLE, 0, i * N, N, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);
    

        // Print the result matrix
    cout << "Result matrix:" << endl;
    printMatrix(C);
    }

    // Free the window object
    MPI_Win_free(&win);

    // Free the matrices
    for (int i = 0; i < N; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
    return 0;
}
