/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program implments the broadcast
 * algorithm using the binomial tree approach.
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int binomialAlgo(int i_size) {
    FILE *f = fopen("binomial_out.txt", "w");

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Request requests[world_size];

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

    for (int i = 0; i < int(log2(world_size)); i++) {
        if (i == 0) {
            if (rank == 0) {
                MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/2), 0, MPI_COMM_WORLD);
            }
            else if (rank == (world_size/2)) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (i == 1) {
            if ((rank % 8) == 0) {
                for (int j = 0; i <= 2; i++) {
                    MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/4), 0, MPI_COMM_WORLD);
                }
            }
            else if ((rank % 4) == 0 && rank > 0) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, world_size/2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_doub = recv_doub;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (i == 1) {
            if ((rank % 8) == 0) {
                for (int j = 0; i <= 4; i++) {
                    MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/8), 0, MPI_COMM_WORLD);
                }
            }
            else if ((rank % 4) == 0 && rank > 0) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 12, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_doub = recv_doub;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (i == 2) {
            if ((rank % 4) == 0) {
                for (int j = 0; i <= 8; i++) {
                    MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/16), 0, MPI_COMM_WORLD);
                }
            }
            else if ((rank % 2) != 0 && rank > 0) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 10, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 12, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 14, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_doub = recv_doub;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (i == 3) {
            if ((rank % 2) == 0) {
                for (int j = 0; i <= 16; i++) {
                    MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/32), 0, MPI_COMM_WORLD);
                }
            }
            else if ((rank % 2) != 0 && rank > 0) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 10, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 12, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 14, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 16, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 18, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 20, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 22, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 24, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 26, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 28, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 30, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_doub = recv_doub;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (i == 4) {
              if ((rank % 2) == 0) {
                for (int j = 0; i <= 32; i++) {
                    MPI_Send(send_doub, arr_size, MPI_DOUBLE, rank+(world_size/32), 0, MPI_COMM_WORLD);
                }
            }
            else if ((rank % 2) != 0 && rank > 0) {
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 6, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 8, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 10, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 12, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 14, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 16, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 18, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 20, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 22, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 24, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 26, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 28, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 30, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 32, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 34, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 36, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 38, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 40, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 42, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 44, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 46, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 48, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 50, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 52, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 54, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 56, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 58, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 60, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_doub, arr_size, MPI_DOUBLE, 62, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                send_doub = recv_doub;
            }
        }
    }

    MPI_Waitall(world_size, requests, MPI_STATUSES_IGNORE);
    
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
        binomialAlgo(i);
    }

    MPI_Finalize();
    return 0;
}
