/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program...
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int pairsPingPong(int i_size) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("World Size: %d ", world_size);
    double start_time, finish_time, run_time;

    // Change this to 0-20 to change size
    int arr_size = pow(2, i_size);
    double *send_doub;
    send_doub = (double *)malloc(sizeof(double)*arr_size);

    // Seed random number generator
    srand(time(NULL));
    // Calculate random double precision variables
    for (int i = 0; i < arr_size; i++) {
        send_doub[i] = (double)(rand()) / RAND_MAX;
    }

    // First pair of ping pong tests
    if (rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
        MPI_Reduce(&run_time, &run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("Size: %d with runtime %f\n", arr_size, run_time);
    }
    else if (rank == 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Second pair of ping pong tests
    if (rank == 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 3, 10, MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 3, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
    }
    else if (rank == 3) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 2, 10, MPI_COMM_WORLD);
    }
}

int syncedPingPong(int i_size) {
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

    if (rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
        printf("Size: %d with runtime %f\n", arr_size, run_time);
    }
    else if (rank == 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int singlePingPong(int i_size) {
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

    if (rank == 0) {
        start_time = MPI_Wtime();
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
        printf("Size: %d with runtime %10f\n", arr_size, run_time);
    }
    else if (rank == 1) {
        MPI_Recv(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(send_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int thousandPingPong() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start_time, finish_time, run_time;
    int arr_size = 1000;
    double send_arr[arr_size];

    if (rank == 0) {
        start_time = MPI_Wtime();
        for (int i = 0; i < 1000; i++) {
            MPI_Send(send_arr, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(send_arr, arr_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        finish_time = MPI_Wtime();
        run_time = finish_time - start_time;
        run_time = run_time / 2000;
        printf("Size: %d with runtime %f\n", arr_size, run_time);
    }
    else if (rank == 1) {
        for (int i = 0; i < 1000; i++) {
            MPI_Recv(send_arr, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_arr, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);

    // thousandPingPong();
    //for (int i = 0; i < 21; i++) {
    //    singlePingPong(i);
    //}

    syncedPingPong(10);
    // pairsPingPong();

    MPI_Finalize();
    return 0;
}