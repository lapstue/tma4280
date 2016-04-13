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


//typedef double double;
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

    double start =  MPI_Wtime();
    double umaxglob=0;

    //printf("Size = %i\n", size);
    // printf("%f\n", start );

    if (argc < 2) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
    }




     if (0 ){

    printf("15 prosent 4 =%i\n", 15%4 );
 }

    // The number of grid points in each direction is n+1
    // The number of degrees of freedom in each direction is n-1
    int n = atoi(argv[1]);
    
    int m = n - 1;  // ant punk hver vei i B

    int cnt[size];
    int displs[size+1];
    displs[size] = m;
    displs[0]=0;




    for(int i = 0;i<size;i++){
        cnt[i] = m / size; // nrColon for hver prosessor  antt elementer jeg eier * ant element den eier
        // if (m % size && i >= (size - m % size)){
        if (m % size && i == 0){
            cnt[i] = cnt[i]+(m%size);
        }
        if (i < size-1){
            displs[i+1] = displs[i]+cnt[i];
        }

                //displs[i] = i * (m / size); //displacement for hver prosessor
    }
    //cnt[size-1] += m%size;

        // if(rank == 0) {
        //     //printf("Prosessor %i har cnt=%i og displs=%i\n", i, cnt[i], displs[i] );
        //     printf("Disp 0 = %i, Disp 1 = %i, Disp 2 = %i, Disp 3 = %i, \n", displs[0], displs[1], displs[2], displs[3]);
        // }

    int nrColon = cnt[rank];
    

   
    int pros_dof = nrColon*m; 

    int nn = 4 * n;
    double h = 1.0 / n;

    int trad = omp_get_max_threads();

    // Grid points
    double *grid = mk_1D_array(n+1, false);

    double **b = mk_2D_array(nrColon, m, false);
    double **bt = mk_2D_array(nrColon, m,false);
    double **z = mk_2D_array(trad,nn, false);

    //double **z = createMatrix(trad, nn);
    double *diag = mk_1D_array(m, false);     

    double *sendbuf = mk_1D_array(nrColon*m, false);
    double *recbuf = mk_1D_array(nrColon*m, false); 
    // int sendcnt = (int)malloc(size*sizeof(int));
    // int sdispls = (int)malloc(size*sizeof(int))

    int sendcnt[size];
    int sdispls[size];

 

    sdispls[0]=0;
    for(int i = 0;i<size;i++){
        sendcnt[i] = cnt[i]*cnt[rank]; //  antt elementer jeg eier * ant element den eier
        sdispls[i] = displs[i]*cnt[rank]; //displacement for hver prosessor
     //   if (rank==1) printf("sdispl=%i  ",sdispls[i]);

    }

 
  // for (int i=1; i<size; i++){
  //   sdispls[i]=sdispls[i-1]+sendcnt[i-1];
  //   if (rank==1) printf("sdispl=%i  ",sdispls[i]);
  // }

   //  printf("Rank=(%i), numCol=%i, sendcnt=%i, sdispls = %i \n",rank, nrColon, sendcnt[1], sdispls[1]);


    // GRID
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n+1; i++) {
        grid[i] = i * h;
         //  printf("%f ", grid[i]);
    }


// int rrr = 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n)); //Stor lamda
   // if (rank == 0)
   //  {
   //    /* code */
   //    printf("Diag[%i] = %f\n", rrr, diag[i] );
   //  }  

    // rrr++;
      }

   //  printf("\n");


    // printf(" Initialize the right hand side data \n" );
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        for (size_t j = 0; j < m; j++) {
          //  b[i][j] = h * h;
             b[i][j] = h * h * func1(grid[i], grid[j]);

        }
    }

    // printf("  Calculate Btilde^T = S^-1 * (S * B)^T \n Bruker hele den FST-greia");
 
    #pragma omp parallel for schedule(guided, 5)
    for (size_t i = 0; i < nrColon; i++) {
        fst_(b[i], &n, z[omp_get_thread_num()], &nn);
    }

    

   // for (size_t i = 0; i < nrColon; i++) {
   //      for (size_t j = 0; j < m; j++) {
   //                 printf("Rank=%i, %f   ",rank, b[i][j]);
   //      }
   //      printf("\n");
   //  }

    


   // transpose(bt, b, m);
    MPItranspose (b, bt,nrColon,m, sendbuf,recbuf,sendcnt,sdispls, size, rank, displs);



// printf("\nETTER\n");




////////////////////////////////////
    //  printf("\nB==\n");
    // for(int i= 0; i<m; i++){
    //     for(int o= 0; o<m; o++){
    //         printf("%f", b[i][o]);
    //     }
    //     printf("\n");
    // }


    // printf("\nBt==\n");
    // for(int i= 0; i<m; i++){
    //     for(int o= 0; o<m; o++){
    //         printf("%f", bt[i][o]);
    //     }
    //     printf("\n");
    // }


    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fstinv_(bt[i], &n, z[omp_get_thread_num()], &nn);
    }



    // printf(" Solve Lambda * Xtilde = Btilde\n");



   // #pragma omp parallel for schedule(static)
    // for (size_t i = 0; i < nrColon; i++) {
    //     for (size_t j = 0; j < m; j++) {
    //         bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
    //  //       printf("%f   ", bt[i][j]);
    //     }
    // }

    #pragma omp parallel for schedule(static)

    for (int j=0; j < nrColon; j++) {
       for (int i=0; i < m; i++) {
            bt[j][i] = bt[j][i]/(diag[i]+diag[j]);
        }
    }


    // for (size_t i = 0; i < nrColon; i++) {
    //     for (size_t j = 0; j < m; j++) {
    //                printf("Rank=%i, %f   ",rank, bt[i][j]);
    //     }
    //     printf("\n");
    // }

    // Calculate X = S^-1 * (S * Xtilde^T)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fst_(bt[i], &n, z[omp_get_thread_num()], &nn);
    }


    //MPItransponse(b,bt, m);

    // printf("\nKOM IHVERTFALL HIT...\n");
    

    

    MPItranspose (bt, b, nrColon,m, sendbuf,recbuf,sendcnt,sdispls, size, rank, displs);


// printf("\nETTER\n");


//  for (size_t i = 0; i < nrColon; i++) {
//         for (size_t j = 0; j < m; j++) {
//                    printf("Rank=%i, %f   ",rank, b[i][j]);
//         }
//         printf("\n");
//     }

     #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nrColon; i++) {
        fstinv_(b[i], &n, z[omp_get_thread_num()], &nn);
    }

    //  for (size_t i = 0; i < nrColon; i++) {
    //     for (size_t j = 0; j < m; j++) {
    //                printf("Rank=%i, %f   ",rank, b[i][j]);
    //     }
    //     printf("\n");
    // }

    // Calculate maximal value of solution
    double u_max = 0.0;


    for (size_t i = 0; i < nrColon; i++) {
       #pragma omp parallel for schedule(static)

        for (size_t j = 0; j < m; j++) {
            u_max = u_max > ( b[i][j] - func2(grid[i], grid[j]) )? u_max : b[i][j];
        }
    }
   // printf("\nMEN IKKE LENGRE...?\n");

    MPI_Reduce (&u_max, &umaxglob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Finalize();

         //     printf("u_max = %e\n", u_max);

     if (rank == 0)
      {
       printf("u_maximus = %e\n", umaxglob);
    double times = MPI_Wtime()-start;
     printf("Tid = %1.16f \n", times);
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


    //MPItranspose (b, bt,nrColon,m, sendbuf,recbuf,sendcnt,sdispls, size, rank, displs);


void MPItranspose(double **b, double **bt, int nrColon, int m, double *sendbuf, double *recbuf, int *sendcnt, int *sdispls, int size, int rank, int *displs ){
    // printf("SENDBUF for rank = %i \n", rank);
    int tt = 0;
    
    for (int o=0; o < size; o++) {

        // printf("Fra prosessor %i Til prosessor %i: ",rank, o );

        for (int i=0; i < nrColon; i++) {
            
            for (int j=displs[o]; j < displs[o+1]; j++) {  
                sendbuf[tt]=b[i][j];

                // printf("%f ", sendbuf[tt]);
                tt++;

            }
       
        }
             // printf(" \n");

    }
    
// printf("REDDI\n"); 

    MPI_Alltoallv(sendbuf, sendcnt, sdispls, MPI_DOUBLE, recbuf, sendcnt, sdispls, MPI_DOUBLE, MPI_COMM_WORLD);

 
    tt = 0;


    for (int o = 0; o < size; o++){
        

        for (int j=displs[o]; j <  displs[o+1]; j++) {
            for (int i=0; i < nrColon; i++) {

                bt[i][j]=recbuf[tt];
               //   printf("DONE"); 

                // printf("%f \n", recbuf[tt]);
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

