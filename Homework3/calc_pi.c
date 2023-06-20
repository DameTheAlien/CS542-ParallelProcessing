/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program...
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// srun --mpi=pmi2 --ntasks=4 --nodes=1 --ntasks-per-node=4 calc_pi.cpp 1000000000

int serialPi(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Please add number of samples\n");
        return 0;
    }

    long n_samples = atol(argv[1]);
    long n_in_circle = 0;
    double rand_x, rand_y;
    double pi;

    // Seed random number generator
    srand(time(NULL));

    // Calculate random x and y values, between 0 and 1
    for (long i = 0; i < n_samples; i++) {
       rand_x = (double)(rand()) / RAND_MAX;  // X is between 0 and 1
       rand_y = (double)(rand()) / RAND_MAX;  // Y is between 0 and 1

       // If inside circle, add to n_in_circle
       if ((rand_x*rand_x) + (rand_y*rand_y) <= 1)
           n_in_circle++;
    }

    // Pi is approximately 4 * number in circle / total number in square
    pi = 4.0*n_in_circle / n_samples;


    printf("Serial: NSamples %ld, Pi Approx %e\n", n_samples, pi);
}

int parallelPi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double start_time, finish_time, run_time;

    long global_n_samples = atol(argv[1]);
    long local_n_samples = atol(argv[1])/world_size;
    long global_n_in_circle = 0;
    long local_n_in_circle = 0;
    double rand_x, rand_y;
    double pi;

    MPI_Barrier(MPI_COMM_WORLD);

    start_time = MPI_Wtime();

    // Seed random number generator
    srand(time(NULL));

    // Calculate random x and y values, between 0 and 1
    for (long i = 0; i < local_n_samples; i++) {
        rand_x = (double)(rand()) / RAND_MAX;  // X is between 0 and 1
        rand_y = (double)(rand()) / RAND_MAX;  // Y is between 0 and 1

        // If inside circle, add to n_in_circle
        if ((rand_x*rand_x) + (rand_y*rand_y) <= 1)
            local_n_in_circle++;
    }

    // MPI reduce to root sum all n_in circle and n_samples
    MPI_Reduce(&local_n_in_circle, &global_n_in_circle, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    finish_time = MPI_Wtime();

    if (rank == 0) {
        // Pi is approximately 4 * number in circle / total number in square
        pi = 4.0*global_n_in_circle / global_n_samples;

        printf("Parallel: NSamples %ld, Pi Approx %e\n", global_n_samples, pi);


        run_time = finish_time - start_time;
        printf("Runtime: %f\n", run_time);
    }
    MPI_Finalize();
}

// Parallel version for approximating pi
int main(int argc, char* argv[]) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    serialPi(argc, argv);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("The elapsed time is %d seconds and %d micros\n", seconds, micros);;

    //start_time = MPI_Wtime();
    parallelPi(argc, argv);
    //finish_time = MPI_Wtime();

    return 0;
}