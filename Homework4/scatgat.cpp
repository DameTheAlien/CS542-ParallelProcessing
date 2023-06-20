/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program implments the broadcast
 * algorithm using the scatter + gather
 * approach.
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int scatgatAlgo(int i_size) {
    FILE *f = fopen("scatgat_out.txt", "w");

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double start_time, finish_time, run_time;

    int arr_size = pow(2, i_size);
    double *send_doub, *recv_doub;
    send_doub = (double *)malloc(sizeof(double)*arr_size);
    recv_doub = (double *)malloc(sizeof(double)*arr_size);

    srand(time(NULL));
    for (int i = 0; i < arr_size; i++) {
        send_doub[i] = (double)(rand()) / RAND_MAX;
    }

    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Scatter first
    MPI_Scatter(send_doub, arr_size, MPI_DOUBLE, recv_doub, arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    recv_doub = send_doub;

    // Ring AllGather second
    for (int i = 0; i < world_size-1; i++) {
        if ((rank % 2) == 0) {
            if (rank == world_size-1) {
                MPI_Send(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
            else {
                MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            }
        }
        else {
            if (rank == 0) {
                MPI_Recv(send_doub, arr_size, MPI_DOUBLE, world_size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(send_doub, arr_size, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    if (rank == 0) {
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
        printf("Size: %d, Time %10f\n", arr_size, run_time);
        fprintf(f, "Size: %d, Time %10f\n", arr_size, run_time);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    for (int i = 0; i < 26; i++) {
        scatgatAlgo(i);
    }

    MPI_Finalize();
    return 0;
}
