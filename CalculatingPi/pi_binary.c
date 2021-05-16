
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

    int tmp_count;
    for(int bin = 2; bin <= size; bin *= 2) {
        if(rank % bin == 0) {
            MPI_Recv(&tmp_count, 1, MPI_INT, rank + (bin / 2), bin, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += tmp_count;
        } 
        else if(rank % bin == bin / 2) {
            MPI_Send(&count, 1, MPI_INT, rank - (bin / 2), bin, MPI_COMM_WORLD);
            break;
        }
    }
    
    if(rank == 0) {
        // Estimate Pi and display the result
        pi = ((double)count / (double)NUM_ITER) * 4.0;

        stop_time = MPI_Wtime();
        
        if(abs(pi - 3.1416) >= 0.01) {
            printf("Wrong pi value: %d\n", pi);
        }
        printf("%d, %f\n", size, stop_time - start_time);
    }

    MPI_Finalize();
    
    return 0;
}

