
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int count = 0;
    double x, y, z, pi;
    
    srand(SEED * rank);

    double start_time, stop_time;
    
    start_time = MPI_Wtime();

    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER / size; iter++) {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0) {
            count++;
        }
    }
    
    if(rank == 0) {
        int tmp_count[size - 1];
        MPI_Request requests[size - 1];
        int global_count = 0;

        for(int i = 1; i < size; ++i) {
            MPI_Irecv(&tmp_count[i - 1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }
        MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);

        global_count += count;
        for(int i = 0; i < size - 1; ++i) {
            global_count += tmp_count[i];
        }
        // Estimate Pi and display the result
        pi = ((double)global_count / (double)NUM_ITER) * 4.0;

        stop_time = MPI_Wtime();
        
        if(abs(pi - 3.1416) >= 0.01) {
            printf("Wrong pi value: %d\n", pi);
        }
        printf("%d, %f\n", size, stop_time - start_time);
    }
    else {
        MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    return 0;
}

