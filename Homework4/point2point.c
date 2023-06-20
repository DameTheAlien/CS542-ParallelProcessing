/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program does a ping pong test on various
 * messages sizes to see how fast point to point
 * communication is in MPI.
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int singlePingPong(int i_size) {
    FILE *f = fopen("pingpong_out.txt", "w");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start_time, finish_time, run_time;

    int arr_size = pow(2, i_size);
    double *send_doub;
    send_doub = (double *)malloc(sizeof(double)*arr_size);

    // Seed random number generator
    srand(time(NULL));
    // Calculate random double precision variable
    for (int i = 0; i < arr_size; i++) {
        send_doub[i] = (double)(rand()) / RAND_MAX;
    }

    start_time = MPI_Wtime();
    if (rank == 0) {
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    finish_time = MPI_Wtime();
    run_time = finish_time - start_time;

    if (rank == 0) {
        printf("Size: %d, Time %10f\n", arr_size, run_time);
        fprintf(f, "Size: %d, Time %10f\n", arr_size, run_time);
    }

    fclose(f);
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    for (int i = 0; i < 16; i++) {
        singlePingPong(i);
    }

    MPI_Finalize();
    return 0;
}