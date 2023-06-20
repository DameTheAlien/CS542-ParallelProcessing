/*
 * Damian Franco
 * CS-542
 * Homework 3
 *
 * This program...
 */
#include "mpi_matmat.hpp"

void mpi_matmat_cannon(double* A, double* B, double* C,
        int n, int sq_num_procs2, int rank_row2, int rank_col2) {
    MPI_Comm cannon_comm;
    MPI_Status status;
    int rank, size;
    int dims[2];
    int periods[2];
    int left, right, up, down;
    double *buf, *tmp;
    int Nl;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    dims[0] = 0;
    dims[1] = 0;
    periods[0] = 1;
    periods[1] = 1;

    MPI_Dims_create(size, 2, dims);

    Nl = n/dims[0];
    buf = (double*)malloc(Nl*Nl*sizeof(double));

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cannon_comm);
    MPI_Cart_shift(cannon_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cannon_comm, 1, 1, &up, &down);

    for(int shift = 0;shift<dims[0]; shift++) {
        for(int i = 0; i < Nl; i++) {
            for(int k = 0; k < Nl; k++) {
                for(int j = 0; j < Nl; j++) {
                    C[i*Nl+j] += A[i*Nl+k]*B[k*Nl+j];
                }
            }
        }


        MPI_Sendrecv(A, Nl*Nl, MPI_DOUBLE, left, 1, buf, Nl*Nl, MPI_DOUBLE, right, 1, cannon_comm, &status);
        tmp = buf;
        buf = A;
        A = tmp;
        MPI_Sendrecv(B, Nl*Nl, MPI_DOUBLE, up, 2, buf, Nl*Nl, MPI_DOUBLE, down, 2, cannon_comm, &status);
        tmp = buf;
        buf = B;
        B = tmp;
    }
}

// Shift A 'rank_row' columns
// Shift B 'rank_col' rows
// All pairs of A and B on a single process should be multiplied
// Then, send submatrix of A to neighboring process (rowwise)
// and submatrix of B to neighboring process (columnwise)
void mpi_matmat_cannon(double* A, double* B, double* C,
       int n, int sq_num_procs, int rank_row, int rank_col)
{
   int rank, num_procs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   int size = n*n;
   int send_proc_A, send_proc_B;
   int recv_proc_A, recv_proc_B;
   int tag_a = 1234;
   int tag_b = 4321;
   MPI_Status status;

   double* send_A = new double[size];
   double* recv_A = new double[size];
   double* send_B = new double[size];
   double* recv_B = new double[size];
   double* tmp;

   for (int i = 0; i < size; i++) {
       C[i] = 0;
   }

   // Determine Send and Recv Processes for Initial Shift
   send_proc_A = get_proc(rank_row, rank_col-rank_row, sq_num_procs);
   send_proc_B = get_proc(rank_row-rank_col, rank_col, sq_num_procs);
   recv_proc_A = get_proc(rank_row, rank_col+rank_row, sq_num_procs);
   recv_proc_B = get_proc(rank_row+rank_col, rank_col, sq_num_procs);

   if (rank_col+rank_row >= sq_num_procs) {
       recv_proc_A = get_proc(rank_row, rank_col+rank_row-sq_num_procs, sq_num_procs);
       recv_proc_B = get_proc(rank_row+rank_col-sq_num_procs, rank_col, sq_num_procs);
   }

   if (rank_col - rank_row < 0) {
       send_proc_A = get_proc(rank_row, rank_col-rank_row+sq_num_procs, sq_num_procs);
   }

   if (rank_row - rank_col < 0) {
       send_proc_B = get_proc(rank_row-rank_col+sq_num_procs, rank_col, sq_num_procs);
   }

   // 1. Perform Initial Shift :
   // Goal : A[rank_row, rank_row+rank_col]*B[rank_row+rank_col, rank_col]
   send_A[rank_row*rank_col] = A[rank_row*rank_row+rank_col];
   MPI_Send(send_A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
   send_B[rank_row*rank_col] = B[rank_row+rank_col*rank_col];
   MPI_Send(send_B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
   MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
   MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);


   // 2. Perform local matrix-multiplication
   // on submatrices received in initial shift
   matmat(n, recv_A, recv_B, C);

   // 3. Determine new values for send_proc_A/B, recv_proc_A/B
   // Make sure to check bounds (wrap around if >= sq_num_procs or < 0)
   // Assign A to [rank_row, rank_col+1]
   send_proc_A = get_proc(rank_row, rank_col+1, sq_num_procs);
   // Assign B to [rank_row+1, rank_col]
   send_proc_B = get_proc(rank_row+1, rank_col, sq_num_procs);
   // Assign A from [rank_row, rank_col-1]
   recv_proc_A = get_proc(rank_row, rank_col-1, sq_num_procs);
   // Assign B from [rank_row-1, rank_col]
   recv_proc_B = get_proc(rank_row-1, rank_col, sq_num_procs);

   // Check wrap arounds and assign wrap around procs
   if (rank_col+1 >= sq_num_procs) {
       send_proc_A = get_proc(rank_row, rank_col+sq_num_procs, sq_num_procs);
   }
   if (rank_row+1 >= sq_num_procs) {
       send_proc_B = get_proc(rank_row+sq_num_procs, rank_col, sq_num_procs);
   }
   if (rank_col-1 < 0) {
       recv_proc_A = get_proc(rank_row, rank_col+sq_num_procs, sq_num_procs);
   }
   if (rank_row-1 < 0) {
       recv_proc_B = get_proc(rank_row+sq_num_procs, rank_col, sq_num_procs);
   }

   // Send A to [rank_row, rank_col+1]
   MPI_Send(send_A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
   // Send B to [rank_row+1, rank_col]
   MPI_Send(send_B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
   // Recv A from [rank_row, rank_col-1]
   MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);
   // Recv B from [rank_row-1, rank_col]
   MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);

   // 4. For each iteration, send and recv A, B, and perform multiplication
   for (int i = 1; i < sq_num_procs; i++)
   {
       // 4a. Send A to send_proc_A
       MPI_Send(send_A, size, MPI_DOUBLE, send_proc_A, tag_a, MPI_COMM_WORLD);
       // 4b. Recv new A from recv_proc_A
       MPI_Recv(recv_A, size, MPI_DOUBLE, recv_proc_A, tag_a, MPI_COMM_WORLD, &status);

       // 4c. Send B to send_proc_B
       MPI_Send(send_B, size, MPI_DOUBLE, send_proc_B, tag_b, MPI_COMM_WORLD);
       // 4c. Recv new B from recv_proc_B
       MPI_Recv(recv_B, size, MPI_DOUBLE, recv_proc_B, tag_b, MPI_COMM_WORLD, &status);


       // 4e. Local matrix multiplication C += recv_A * recv_B
       matmat(n, recv_A, recv_B, C);
   }

   delete[] send_A;
   delete[] recv_A;
   delete[] send_B;
   delete[] recv_B;
}