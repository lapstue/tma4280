/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015

 * Paralellisert av Peter Holiman og Kåre Birger Lapstuen
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


typedef int bool;

// Function prototypes
double *mk_1D_array(size_t n, bool zero);
double **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(double **bt, double **b, size_t m);
void MPItranspose(double **b, double **bt, int nrColon, int m, double *sendbuf, double *recbuf, int *sendcnt, int *sdispls, int size, int rank, int *displs );
double func1(double x, double y);
double func2(double x, double y);
void fst_(double *v, int *n, double *w, int *nn);
void fstinv_(double *v, int *n, double *w, int *nn);




int main(int argc, char **argv)
{

    int size , rank, number; 

    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD , &size);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);

    double start =  MPI_Wtime(); //Starter klokka
    double umaxglob=0; //Max feil for alle tråder

    if (argc < 2) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
    }

    // The number of grid points in each direction is n+1
    // The number of degrees of freedom in each direction is n-1
    int n = atoi(argv[1]);
    
    int m = n - 1;  // ant punk hver vei i B

    int *cnt = (int *) malloc(size * sizeof(int)); //loakal ant kolonner i matrix
    int *displs = (int *) malloc((size+1) * sizeof(int)); //lokal displacement for de andre prosessorene sine punkter i sendbuf
    displs[size] = m;
    displs[0]=0; //Displacement til første prosessor er alltid 0




    for(int i = 0;i<size;i++){
        cnt[i] = m / size; // nrColon for hver prosessor  
        if (m % size && i == 0){ //prosess 0 får resten om det ikke er delelig
            cnt[i] = cnt[i]+(m%size);
        }
        if (i < size-1){
            displs[i+1] = displs[i]+cnt[i];
        }

    }
 

    int nrColon = cnt[rank]; //ant kolonner "jeg" har
    int pros_dof = nrColon*m;  //ant lementer jeg har

    int nn = 4 * n;
    double h = 1.0 / n;


    // Grid points
    double *grid = mk_1D_array(n+1, false);
    double **b = mk_2D_array(nrColon, m, false);
    double **bt = mk_2D_array(nrColon, m,false);

    int trad = omp_get_max_threads(); //ant tråder 
    double **z = mk_2D_array(trad,nn, false); //z er 2D pga paralellisering med OpenMP, da FST ikke skal overskrive andre tråders z

    double *diag = mk_1D_array(m, false);     
    double *sendbuf = mk_1D_array(nrColon*m, false);
    double *recbuf = mk_1D_array(nrColon*m, false); 


    int *sendcnt = (int *) malloc((size+1) * sizeof(int)); //ant elementer jeg skal sende hver prosessor 
    int *sdispls = (int *) malloc((size+1) * sizeof(int)); //index i sendbuf for hver prosessor

 

    sdispls[0]=0; //prosessor 0 skal alltid ha fra index 0
    for(int i = 0;i<size;i++){
        sendcnt[i] = cnt[i]*cnt[rank]; //  antt elementer jeg eier * ant element den eier
        sdispls[i] = displs[i]*cnt[rank]; //displacement for hver prosessor
    }

    // GRID
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n+1; i++) {
        grid[i] = i * h;
    }


    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n)); //Eigenvalue
      }

  // Initialize the right hand side data 
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        for (size_t j = 0; j < m; j++) {
        //  b[i][j] = h * h;
            b[i][j] = h * h * func1(grid[i], grid[j]); //evaluerer funksjoen * h*h
        }
    }

    // Calculate Btilde^T = S^-1 * (S * B)^T 
 
    #pragma omp parallel for schedule(guided, 5)
    for (size_t i = 0; i < nrColon; i++) {
        fst_(b[i], &n, z[omp_get_thread_num()], &nn);
    }
    MPItranspose (b, bt,nrColon,m, sendbuf,recbuf,sendcnt,sdispls, size, rank, displs);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fstinv_(bt[i], &n, z[omp_get_thread_num()], &nn);
    }

    // Solve Lambda * Xtilde = Btilde

    #pragma omp parallel for schedule(static)

    for (int j=0; j < nrColon; j++) {
       for (int i=0; i < m; i++) {
            bt[j][i] = bt[j][i]/(diag[i]+diag[j]);
        }
    }

    // Calculate X = S^-1 * (S * Xtilde^T)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fst_(bt[i], &n, z[omp_get_thread_num()], &nn);

    }
    MPItranspose (bt, b, nrColon,m, sendbuf,recbuf,sendcnt,sdispls, size, rank, displs);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fstinv_(b[i], &n, z[omp_get_thread_num()], &nn);
    }

    // Calculate maximal value of solution
    double u_max = 0.0;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        for (size_t j = 0; j < m; j++) {
            u_max = u_max > ( b[i][j] - func2(grid[i], grid[j]) )? u_max : b[i][j]; //tester resultat - kjent funksjon, skal bli = 0
        }
    }
    MPI_Reduce (&u_max, &umaxglob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); //Finner den største u_max fra de forskjellige prosessorene og setter den til umaxglob 

    MPI_Finalize();

    if (rank == 0) {
        printf("Nodes = %d \n", size);
        printf("Threads per node = %d \n", omp_get_max_threads());
        printf("u_max = %e\n", umaxglob);  //Printer max feil
        double times = MPI_Wtime()-start; //Stopper klokka
        printf("Time elapsed = %1.16f \n", times); //Pinter tid
    }
    

    return 0;
}

double func1(double x, double y) {
    //return 2 * (y - y*y + x - x*x);
    return 5.0*PI*PI*sin(PI*x)*sin(2.0*PI*y);

}

double func2(double x, double y) {
    //return 2 * (y - y*y + x - x*x);
    return sin(PI*x)*sin(2.0*PI*y);

}

void transpose(double **bt, double **b, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
           
            bt[i][j] = b[j][i];
        }
    }
}



void MPItranspose(double **b, double **bt, int nrColon, int m, double *sendbuf, double *recbuf, int *sendcnt, int *sdispls, int size, int rank, int *displs ){
    int tt = 0; //teller
    
    for (int o=0; o < size; o++) { //går igjennom hver prosessor
        for (int i=0; i < nrColon; i++) { 
            for (int j=displs[o]; j < displs[o+1]; j++) {  //går igjennom det som skal sendes tl prosessoren med rank=  o
                sendbuf[tt]=b[i][j]; //fyller sendbuf 
                tt++;
            }
        }
    }    
    MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPI_DOUBLE, recbuf, sendcnt, sdispls, MPI_DOUBLE, MPI_COMM_WORLD);
    //Sender til alle prosessorer
    tt = 0;
    for (int o = 0; o < size; o++){ //går igjennom hver prosessor
        for (int j=displs[o]; j <  displs[o+1]; j++) { //Tar displacementen først for å også da transponere selve innholdet
            for (int i=0; i < nrColon; i++) {
                bt[i][j]=recbuf[tt]; //Skriver til bt
                tt++;
            }
        }
    }
}



double *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (double *)calloc(n, sizeof(double));
    }
    return (double *)malloc(n * sizeof(double));
}

double **mk_2D_array(size_t n1, size_t n2, bool zero)
{
    double **ret = (double **)malloc(n1 * sizeof(double *));

    if (zero) {
        ret[0] = (double *)calloc(n1 * n2, sizeof(double));
    }
    else {
        ret[0] = (double *)malloc(n1 * n2 * sizeof(double));
    }

    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

