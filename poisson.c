/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <mpi.h>
#include <omp.h>

#define PI 3.14159265358979323846
#define true 1
#define false 0

//B i vår kode er samme som B i forelesning


typedef double real;
typedef int bool;

// Function prototypes
real *mk_1D_array(size_t n, bool zero);
real **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(real **bt, real **b, size_t m);
real rhs(real x, real y);
void fst_(real *v, int *n, real *w, int *nn);
void fstinv_(real *v, int *n, real *w, int *nn);

int main(int argc, char **argv)
{

    int size , rank, number; 
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD , &size);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);

    double start =  omp_get_wtime();
    printf("%f\n", start );

    if (argc < 2) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
    }

    //int nproc = atoi(argv[2]);

    if(rank == 0){
        printf("Prosess 1\n");
    }

    if(rank != 0){
        printf("Prosess %d \n", rank);
    }

    
    MPI_Finalize();

    // The number of grid points in each direction is n+1
    // The number of degrees of freedom in each direction is n-1
    int n = atoi(argv[1]);
    int m = n - 1;
    int nn = 4 * n;
    real h = 1.0 / n;

    // Grid points
    real *grid = mk_1D_array(n+1, false);

    printf("GRID\n");
    for (size_t i = 0; i < n+1; i++) {
        grid[i] = i * h;
         //  printf("%f ", grid[i]);
    }
 //   printf("\n");



        printf("The diagonal of the eigenvalue matrix of T\n");

    real *diag = mk_1D_array(m, false);
    for (size_t i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n)); //Stor lamda
      //  printf("%f   ", diag[i]);
    }

   //  printf("\n");


    printf(" Initialize the right hand side data \n" );
    real **b = mk_2D_array(m, m, false);
    real **bt = mk_2D_array(m, m, false);
    real *z = mk_1D_array(nn, false);

    printf("Z==\n");
    for(int i= 0; i<nn; i++){
        printf("%d\n", z[i]);
    }
    printf("\n");

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
          //FLER TRÅDER UTEN PROB
            b[i][j] = h * h * rhs(grid[i], grid[j]);
             //       printf("%f   ", b[i][j]);

        }
      //  printf("\n");
    }

    printf("  Calculate Btilde^T = S^-1 * (S * B)^T \n Bruker hele den FST-greia");
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        fst_(b[i], &n, z, &nn);
    }
    transpose(bt, b, m);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        fstinv_(bt[i], &n, z, &nn);
    }

    printf(" Solve Lambda * Xtilde = Btilde\n");



    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
     //       printf("%f   ", bt[i][j]);
        }
    }

    // Calculate X = S^-1 * (S * Xtilde^T)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        fst_(bt[i], &n, z, &nn);
    }
    //MPI
    transpose(b, bt, m);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        fstinv_(b[i], &n, z, &nn);
    }

    // Calculate maximal value of solution
    double u_max = 0.0;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            u_max = u_max > b[i][j] ? u_max : b[i][j];
        }
    }

    printf("u_maximus = %e\n", u_max);

    double times = omp_get_wtime()-start;
    printf("Tid = %1.16f \n", times);

    return 0;
}

real rhs(real x, real y) {
    return 2 * (y - y*y + x - x*x);
}

void transpose(real **bt, real **b, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
           
            bt[i][j] = b[j][i];
        }
    }
}

real *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (real *)calloc(n, sizeof(real));
    }
    return (real *)malloc(n * sizeof(real));
}

real **mk_2D_array(size_t n1, size_t n2, bool zero)
{
    real **ret = (real **)malloc(n1 * sizeof(real *));

    if (zero) {
        ret[0] = (real *)calloc(n1 * n2, sizeof(real));
    }
    else {
        ret[0] = (real *)malloc(n1 * n2 * sizeof(real));
    }

    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

