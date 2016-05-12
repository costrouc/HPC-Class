#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <cublas.h>

#include "timing.cpp"


#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

//=============================================================================
// Standard CG routine in double precision arithmetic
//=============================================================================

// Reference SpMV product on the PCU
void dsmv(double *h_A, int *h_I, int *h_J, int N, double *h_X, double *h_Y)
{
   double res;
   for(int i=0; i<N; i++)
   {
      res=0;

        for(int j=h_I[i];j<h_I[i+1];j++)
        {
           res+=h_A[j]*h_X[h_J[j]];
        }
      h_Y[i]=res;
   }
}


void CGd(int dofs, int & num_of_iter,  double *x, double *b,
         double *h_A, int *h_I, int *h_J,
         double rtol = RTOLERANCE ){
  double *r=new double[dofs], *d=new double[dofs], *z=new double[dofs];
  double r0, den, nom, nom0, betanom, alpha, beta;
  int i, j;

  nom = 0.0;
  for(j=0; j<dofs; j++){
    x[j] = 0.;
    r[j] = d[j] = b[j];

    nom += r[j]*r[j];
  }
  nom0 = nom;                                 // nom = r dot r
  dsmv(h_A, h_I, h_J, dofs, r, z);            //   z = A r
  den = 0.0;

  for(j=0; j<dofs; j++)
    den += z[j]*r[j];                         // den = z dot r

  if ( (r0 = nom * rtol) < ATOLERANCE) r0 = ATOLERANCE;
  if (nom < r0)
    return;

  if (den <= 0.0) {
    printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
    return;
  }

  // printf("Iteration : %4d  Norm: %f\n", 0, nom);

  // start iteration
  for(i= 1; i<num_of_iter ;i++) {
    alpha = nom/den;

    betanom = 0.0;
    for(j=0;j<dofs; j++){
      x[j] += alpha*d[j];                         //  x = x + alpha d
      r[j] -= alpha*z[j];                         //  r = r - alpha z
      betanom += r[j]*r[j];                       //  betanom = r dot r
    }

    // printf("Iteration : %4d  Norm: %f\n", i, betanom);
    if ( betanom < r0 ) {
      num_of_iter = i;
      break;
    }

    beta = betanom/nom;                           // beta = betanom/nom
    for(j=0;j<dofs; j++)
      d[j] = r[j] + beta * d[j];                  // d = r + beta d           

    dsmv(h_A, h_I, h_J, dofs, d, z);              // z = A d
    den = 0.;
    for(j=0;j<dofs; j++)
      den += d[j]*z[j];                           // den = d dot z
    nom = betanom;
  } // end iteration

  printf( "      (r_0, r_0) = %e\n", nom0);
  printf( "      (r_N, r_N) = %e\n", betanom);
  printf( "      Number of CG iterations: %d\n", i);

  if (rtol == RTOLERANCE) {
    dsmv(h_A, h_I, h_J, dofs, x, r);              //    r = A x
    den = 0.0;
    for(j=0; j<dofs; j++){
      r[j] = b[j] - r[j];                         //    r = b  - r
      den += r[j]*r[j];
    }
    printf( "      || r_N ||   = %f\n", sqrt(den));
  }

  delete [] r;
  delete [] z;
  delete [] d;
}


//=============================================================================
// Standard CG routine in double precision arithmetic on the GPU
//=============================================================================

// SpMV on the GPU
#define num_threads 32
__global__ void dsmv_kernel(double* A, int *I, int *J, int n, double *d_X, double *d_Y)
{
   int ind = blockIdx.x*num_threads + threadIdx.x;

   if (ind < n){
      I += ind;
 
      int j, last=I[1];
      double res = 0.f;

      for(j=I[0];j<last;j++)
         res += A[j] * d_X[ J[j] ];

      d_Y[ind] = res;
   }
}


void dsmv_gpu(double *d_A, int *d_I, int *d_J, int N, double *d_X, double *d_Y)
{
   dim3 grid(N/num_threads, 1, 1);
   dim3 threads(num_threads, 1, 1);

   dsmv_kernel<<<grid, threads>>>(d_A, d_I, d_J, N, d_X, d_Y);
}


void CGd_GPU(int dofs, int & num_of_iter,  double *x, double *b,
             double *d_A, int *d_I, int *d_J, double *dwork,
             double rtol = RTOLERANCE ){

  double *r = dwork;
  double *d = dwork + dofs;
  double *z = dwork + 2*dofs;

  double r0, den, nom, nom0, betanom, alpha, beta;
  int i;

  cublasDscal(dofs, 0.f, x, 1);        // x = 0
  cublasDcopy(dofs, b, 1, r, 1);       // r = b
  cublasDcopy(dofs, b, 1, d, 1);       // d = b
  nom = cublasDnrm2(dofs, r, 1);       // nom = || r ||
  nom = nom * nom;

  nom0 = nom;                          // nom = r dot r
  dsmv_gpu(d_A, d_I, d_J, dofs, r, z); // z = A r
  den = cublasDdot(dofs, z, 1, r, 1);  // den = z dot r

  if ( (r0 = nom * rtol) < ATOLERANCE) r0 = ATOLERANCE;
  if (nom < r0)
    return;

  if (den <= 0.0) {
    printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
    return;
  }

  // printf("Iteration : %4d  Norm: %f\n", 0, nom);

  // start iteration
  for(i= 1; i<num_of_iter ;i++) {
    alpha = nom/den;
    cublasDaxpy(dofs,  alpha, d, 1, x, 1);         // x = x + alpha d
    cublasDaxpy(dofs, -alpha, z, 1, r, 1);         // r = r - alpha z
    betanom = cublasDnrm2(dofs, r, 1);             // betanom = || r ||
    betanom = betanom * betanom;                   // betanom = r dot r

    // printf("Iteration : %4d  Norm: %f\n", i, betanom);
    if ( betanom < r0 ) {
      num_of_iter = i;
      break;
    }

    beta = betanom/nom;                           // beta = betanom/nom
    cublasDscal(dofs, beta, d, 1);                // d = beta*d
    cublasDaxpy(dofs, 1.f, r, 1, d, 1);           // d = d + r
    dsmv_gpu(d_A, d_I, d_J, dofs, d, z);          // z = A d
    den = cublasDdot(dofs, d, 1, z, 1);           // den = d dot z

    nom = betanom;
  } // end iteration

  printf( "      (r_0, r_0) = %e\n", nom0);
  printf( "      (r_N, r_N) = %e\n", betanom);
  printf( "      Number of CG iterations: %d\n", i);

  if (rtol == RTOLERANCE) {
    dsmv_gpu(d_A, d_I, d_J, dofs, x, r);          // r = A x
    cublasDaxpy(dofs,  -1.f, b, 1, r, 1);         // r = r - b
    den = cublasDnrm2(dofs, r, 1);                // den = || r ||
    printf( "      || r_N ||   = %f\n", den);
  }
}
//============================================================================

int main(int argc,char **argv)
{

   cuInit( 0 );
   cublasInit( );

   TimeStruct start, end;

   int N, i, NNZ, inc=0, filelines=0;
   int read,col1,row1;
   float val1;
   FILE   *DataFile;

   //======================Reading file=======================================
   //==========================================================================

   printf("\n....... Reading matrix.output ......................... \n");

   if ((DataFile = fopen("matrix.output", "r")) == NULL)
      printf("\nCan't read matrix.output\n");

   fscanf(DataFile,"%d%d%d", &N, &N, &NNZ);

   int current_col = 0, k = 0, *nnz_row;
   nnz_row   = (int*)malloc( sizeof(int)*(N+1));
   nnz_row[k] = inc;
   //=======================Memory allocation=================
   //===========================================================

   double *h_Y, *h_X, *d_X, *d_Y, *h_Y1, *dwork;
   double *h_A,*d_A;
   int *h_J, *h_I, *d_J, *d_I;

   h_X=(double*)malloc(N*sizeof(double)); 
   if (h_X==NULL) printf("fail to allocate h_X\n"), exit(1);
  
   h_A=(double*)malloc((NNZ+1)*sizeof(double));
   if (h_A==NULL) printf("fail to allocate h_A\n"), exit(1);
    
   h_Y=(double*)malloc(N*sizeof(double));
   if (h_Y==NULL) printf("fail to allocate h_Y\n"), exit(1);
    
   h_Y1=(double*)malloc(N*sizeof(double));
   if (h_Y1==NULL) printf("fail to allocate h_Y1\n"), exit(1);
    
   h_A=(double*)malloc(NNZ*sizeof(double)); 
   if (h_A==NULL) printf("fail to allocate h_A\n"), exit(1);
    
   h_J=(int*)malloc((NNZ+1)*sizeof(int));    
   if (h_J==NULL) printf("fail to allocate h_J\n"), exit(1);
    
   h_I=(int*)malloc((N+1)*sizeof(int));  
   if (h_I==NULL) printf("fail to allocate h_I\n"), exit(1);

   for(i=0; i<N; i++)
     h_X[i] = 1.f*rand()/RAND_MAX;

   for(i=0;i<NNZ;i++){
      read=fscanf(DataFile,"%d%d%f",&col1,&row1,&val1);

      if(read!=3)break;
      h_J[filelines]=col1-1;

      if (current_col == row1-1)
        nnz_row[k]++;
      else
      {
        current_col = row1-1;
        k++;
        inc=1;
        nnz_row[k] = 1;
      }

      h_A[filelines]=val1;

      filelines++;
   }
   printf("file lines: %d\n", filelines);

   fclose(DataFile);
   fprintf(stderr,"File is closed\n");
 
   h_I[0]=0;
   i = 0;
   for(i=1;i<=N;i++)
     h_I[i]=h_I[i-1] + nnz_row[i-1];

   printf("N = %d\n", N);
   //===========================================================
   //=============sparse Matrix vector product on CPU================
   //==============================================================
   start = get_current_time();
   dsmv(h_A, h_I, h_J, N, h_X, h_Y);          // h_Y = h_A * h_X
   end = get_current_time();

   printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
   printf("Speed: %f GFlops \n", 2.*NNZ/
           (1.*1000000*GetTimerValue(start,end)));

   //======================================================
   //=====================GPU=============================
   //===================================================

   printf("....... allocating GPU memory ........................... \n\n");

   cudaMalloc((void**)&dwork,3*N*sizeof(double));
   cudaMalloc((void**)&d_X,N*sizeof(double));
   cudaMalloc((void**)&d_Y,N*sizeof(double));
   cudaMalloc((void**)&d_A,(NNZ+1)*sizeof(double));
   cudaMalloc((void**)&d_I,(N+1)*sizeof(int));
   cudaMalloc((void**)&d_J,(NNZ+1)*sizeof(int));

   cudaMemcpy(d_A, h_A, NNZ*sizeof(double),    cudaMemcpyHostToDevice);
   cudaMemcpy(d_J, h_J, (NNZ+1)*sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpy(d_I, h_I, (N+1)*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(d_X, h_X, N*sizeof(double),      cudaMemcpyHostToDevice);

   printf("memory allocated\n");

   start = get_current_time();
   dsmv_gpu(d_A, d_I, d_J, N, d_X, d_Y);
   end = get_current_time();

   cudaMemcpy(h_Y1, d_Y, N*sizeof(double), cudaMemcpyDeviceToHost);

   //==========================================================================
   //=======print the result( first three values )from GPU=====================
   printf("\n....................................................... \n");
   double norm = 0.f;
   for(i=0;i<N;i++)
     norm += (h_Y[i] - h_Y1[i])*(h_Y[i] - h_Y1[i]);

   //==========================================================================
   printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
   printf("Speed: %f GFlops \n", 2.*NNZ/
           (1.*1000000*GetTimerValue(start, end)));
   printf("|| Y_GPU - Y_CPU ||_2 = %f \n", sqrt(norm));

   //==========================================================================
   // Solve  h_A * h_X = h_Y on the CPU using CG
   int max_num_iters = 5000;
   printf("\n....... Solving Ax = b using CG on the CPU ............ \n");

   start = get_current_time();
   CGd( N, max_num_iters, h_X, h_Y,
        h_A, h_I, h_J);
   end = get_current_time();

   printf("Time (s) = %f\n", GetTimerValue(start,end)/1000.);

   //==========================================================================
   // Solve  d_A * d_X = d_Y on the GPU using CG
   printf("\n....... Solving Ax = b using CG on the GPU ............ \n");

   start = get_current_time();
   CGd_GPU(N, max_num_iters, d_X, d_Y, d_A, d_I, d_J, dwork);
   end = get_current_time();

   printf("Time (s) = %f\n\n", GetTimerValue(start,end)/1000.);

   cudaFree(d_X);
   cudaFree(d_Y);
   cudaFree(d_A);
   cudaFree(d_I);
   cudaFree(d_J);
   cudaFree(dwork);

   free(h_A);
   free(h_X);
   free(h_Y);
   free(h_Y1);
   free(h_J);
   free(h_I);
}
