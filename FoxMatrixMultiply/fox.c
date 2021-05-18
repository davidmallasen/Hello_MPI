#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEBUG 0

// Based on: http://csis.uni-wuppertal.de/courses/scripts/lab2_chapters/PP4.pdf
typedef struct {
    int size;  // Number of processes
    MPI_Comm comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int side_len;  // Side length of the square of processes
    int my_row;
    int my_col;
    int my_rank;
} proc_info_t;


// Fox algorithm functions
double* fox(double *mat_a_block, double *mat_b_block, int mat_dim, int local_dim, 
            proc_info_t *grid);
void broadcast_diag_row(double *bcast_a_block, double *mat_a_block, int local_dim, int step, 
                        proc_info_t* grid);
void local_dotp(double *acc, double *a, double *b, int n);

// Auxiliary functions
void setup_grid(proc_info_t* grid);
void read_matrix(double *local_mat, int local_dim, int mat_dim, proc_info_t *grid);
void write_matrix(double *local_mat, int local_dim, int mat_dim, proc_info_t *grid);


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    proc_info_t grid;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    setup_grid(&grid);

#if DEBUG
    int local_dim = 2;
#else
    int local_dim = 1000; 
#endif
    int mat_dim = grid.side_len * local_dim;

    double *mat_a_block = (double*) malloc(local_dim * local_dim * sizeof(double));
    double *mat_b_block = (double*) malloc(local_dim * local_dim * sizeof(double));
    double *mat_c_block;

    read_matrix(mat_a_block, local_dim, mat_dim, &grid);
    read_matrix(mat_b_block, local_dim, mat_dim, &grid);

    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    mat_c_block = fox(mat_a_block, mat_b_block, mat_dim, local_dim, &grid);

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

#if DEBUG
    write_matrix(mat_c_block, local_dim, mat_dim, &grid);
#endif

    if (grid.my_rank == 0) {
        printf("Time elapsed: %lf\n", elapsed_time);
    }

    free(mat_a_block);
    free(mat_b_block);
    free(mat_c_block);

    MPI_Finalize();

    return 0;
}


double* fox(double *mat_a_block, double *mat_b_block, int mat_dim, int local_dim, 
            proc_info_t *grid) {
    double *bcast_a_block = (double*) malloc(local_dim * local_dim * sizeof(double)); 
    double *mat_c_block   = (double*) calloc(local_dim * local_dim, sizeof(double));

    // Processes above and below
    int source = (grid->my_row + 1 ) % grid->side_len;
    int dest   = (grid->my_row - 1 + grid->side_len) % grid->side_len;

    for (int i = 0; i < grid->side_len; i++) {
        // Broadcast diagonal to all the row
        broadcast_diag_row(bcast_a_block, mat_a_block, local_dim, i, grid);

        // Matmul
        local_dotp(mat_c_block, bcast_a_block, mat_b_block, local_dim);

        // Roll B block
        MPI_Sendrecv_replace(mat_b_block, local_dim * local_dim, MPI_DOUBLE, dest, 0, source, 0, 
                             grid->col_comm, MPI_STATUS_IGNORE);
    }

    free(bcast_a_block);

    return mat_c_block;
}

void broadcast_diag_row(double *bcast_a_block, double *mat_a_block, int local_dim, int step, 
                        proc_info_t* grid) {
    // Process in each row that must broadcast its A block
    int row_sender = (grid->my_row + step) % grid->side_len;

    if (grid->my_col == row_sender) {
        memcpy(bcast_a_block, mat_a_block, local_dim * local_dim * sizeof(double));
    }
    
    // Broadcast A block to the whole row
    MPI_Bcast(bcast_a_block, local_dim * local_dim, MPI_DOUBLE, row_sender, grid->row_comm);
}

void local_dotp(double *acc, double *a, double *b, int n) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                acc[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}


void setup_grid(proc_info_t* grid) {
    // Fill the sizes
    MPI_Comm_size(MPI_COMM_WORLD, &grid->size);
    grid->side_len = (int) sqrt((double) grid->size);

    // Using cartesian topology
    int dim_sizes[2];
    int wrap_around[2];

    dim_sizes[0] = grid->side_len;
    dim_sizes[1] = grid->side_len;
    wrap_around[0] = 1;
    wrap_around[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, 1, &grid->comm);

    int coordinates[2];

    // reorder=1 in MPI_Cart_create could change the rank before
    MPI_Comm_rank(grid->comm, &grid->my_rank);

    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0]; // Row of the process, Y value
    grid->my_col = coordinates[1]; // Column of the process, X value

    // Partition grid into rows and columns
    int free_coords[2];

    free_coords[0] = 0; // Row coordinate fixed
    free_coords[1] = 1; // Column coordinate free
    MPI_Cart_sub(grid->comm, free_coords, &grid->row_comm);

    free_coords[0] = 1; // Row coordinate free
    free_coords[1] = 0; // Column coordinate fixed
    MPI_Cart_sub(grid->comm, free_coords, &grid->col_comm);
}

void read_matrix(double *local_mat, int local_dim, int mat_dim, proc_info_t *grid) {
    int dest;
    int coords[2];

    // Process 0 reads all the matrix and sends the blocks to the other
    //  processes
    if (grid->my_rank == 0) {
        double* temp_row = (double*) malloc(local_dim * sizeof(double));

        for (int mat_row = 0; mat_row < mat_dim; mat_row++) {
            // Local row corresponding to current column pointer
            coords[0] = mat_row / local_dim;

            for (int grid_col = 0; grid_col < local_dim; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (int mat_col = 0; mat_col < local_dim; mat_col++) {
#if DEBUG
                        scanf("%lf", &local_mat[mat_row * local_dim + mat_col]);
#else
                        local_mat[mat_row * local_dim + mat_col] = (1 / (double)RAND_MAX) * rand();
#endif
                    }
                }
                else {
                    for (int mat_col = 0; mat_col < local_dim; mat_col++) {
#if DEBUG
                        scanf("%lf", &temp_row[mat_col]);
#else
                        temp_row[mat_col] = (1 / (double)RAND_MAX) * rand();
#endif
                    }
                    MPI_Send(temp_row, local_dim, MPI_DOUBLE, dest, 0, grid->comm);
                }
            }
        }

        free(temp_row);
    }
    else {  // Receive matrix block row by row from process 0
        for (int mat_row = 0; mat_row < local_dim; mat_row++) {
            MPI_Recv(&local_mat[mat_row * local_dim], local_dim, MPI_DOUBLE, 0, 0, grid->comm,
                     MPI_STATUS_IGNORE);
        }
    }
}

void write_matrix(double *local_mat, int local_dim, int mat_dim, 
                  proc_info_t *grid) {
    int source;
    int coords[2];

    // Process 0 receives all the matrix blocks and writes them
    if (grid->my_rank == 0) {
        double* temp_row = (double*) malloc(local_dim * sizeof(double));

        for (int mat_row = 0; mat_row < mat_dim; mat_row++) {
            // Local row corresponding to current column pointer
            coords[0] = mat_row / local_dim;

            for (int grid_col = 0; grid_col < local_dim; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(int mat_col = 0; mat_col < local_dim; mat_col++) {
                        printf("%lf ", local_mat[mat_row * local_dim + mat_col]);
                    }
                }
                else {
                    MPI_Recv(temp_row, local_dim, MPI_DOUBLE, source, 0, grid->comm,
                             MPI_STATUS_IGNORE);
                    for(int mat_col = 0; mat_col < local_dim; mat_col++) {
                        printf("%lf ", temp_row[mat_col]);
                    }
                }
            }
            printf("\n");
        }

        free(temp_row);
    } 
    else { // Send matrix block row by row to process 0
        for (int mat_row = 0; mat_row < local_dim; mat_row++) {
            MPI_Send(&local_mat[mat_row * local_dim], local_dim, MPI_DOUBLE, 0, 0, grid->comm);
        }
    }
}
