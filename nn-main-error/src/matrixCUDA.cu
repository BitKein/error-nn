#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "nn_aux.h"
#include "globals.h"
#include "../include/matrixCUDA.cuh"

#define THR_PER_BLOCK 1024

//double **alloc_matrix_2v_CUDA(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)){}

//double **alloc_matrix_1v_CUDA(int n_layers, int *size, double (*init_weight_ptr)(void)){}

//double *alloc_array_CUDA(int length){}

//double *alloc_matrix_CUDA(int rows, int cols){}

//double *m_elem_CUDA(double *m, int length, int x, int y){}


//verified
__global__ void GPU_matrix_sum(double* A, double* B, double* C, int rows, int cols){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows*cols){
        C[i] = A[i] + B[i];
    }
}

//verified
void matrix_sum_CUDA(double *c, double *a, double *b, int rows, int cols){
    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, a, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, b, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_sum<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, rows,cols);
    gpuErrchk(cudaEventRecord(stop));

   // Copy data from device array d_C to host array C
    cudaMemcpy(c, d_C, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


//verified
__global__ void GPU_matrix_sub(double *C, double *A, double *B, int rows, int cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows*cols)
        C[i] = A[i] - B[i];
    
}

//verified
void matrix_sub_CUDA(double *c, double *a, double *b, int rows, int cols){
    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, a, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, b, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_sub<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, rows,cols);
    gpuErrchk(cudaEventRecord(stop));

   // Copy data from device array d_C to host array C
    cudaMemcpy(c, d_C, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


//verified
__global__ void GPU_matrix_mul_cnt(double *m, int rows, int cols, double cnt){
    int d=blockIdx.x * blockDim.x + threadIdx.x;
    if(d<rows*cols){
        *(m+d) *= cnt;
    }
}

//verified
//multiplicar cada elemento de una matriz por un número cnt
void matrix_mul_cnt_CUDA(double *m, int rows, int cols, double cnt){
     cudaEvent_t start, stop;
    double *d_A;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, rows*cols * sizeof(double)));
    //gpuErrchk(cudaMalloc(&d_B, rows*cols * sizeof(double)));
    //gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, m, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_B, b, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_mul_cnt<<<blk_in_grid, thr_per_blk>>>(d_A, rows,cols,cnt);
    gpuErrchk(cudaEventRecord(stop));

   // Copy data from device array d_C to host array C
    cudaMemcpy(m, d_A, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);

}

//verified
__global__ void GPU_matrix_zero(double *m, int rows, int cols){
    int d=blockIdx.x * blockDim.x + threadIdx.x;
    if(d<rows*cols){
        *(m+d) = 0.0;
    }
}

//verified
//poner a 0.0 los elementos de una matriz
void matrix_zero_CUDA(double *m, int rows, int cols){
    double *d_m;

    cudaMalloc(&d_m,cols*rows*sizeof(double));
    

    cudaMemcpy(d_m,m,cols*rows*sizeof(double),cudaMemcpyHostToDevice);


    int thr_per_blk=cols;
    int blk_in_grid=rows;
    GPU_matrix_zero<<<blk_in_grid,thr_per_blk>>>(d_m,rows,cols);

    cudaMemcpy(m,d_m,cols*rows*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(d_m);
}


//verified
__global__ void GPU_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){
    int d=blockIdx.x * blockDim.x + threadIdx.x;
    if(d<rows*cols){
        int d=blockIdx.x * blockDim.x + threadIdx.x;
        if(d<rows*cols){
            *(c+d)=*(a+d) * *(b+d);
        }
    }
}

//verified
void matrix_mul_dot_CUDA(double *c, double *a, double *b, int rows, int cols){
     cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, a, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, b, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_mul_dot<<<blk_in_grid, thr_per_blk>>>(d_A,d_B,d_C,rows,cols);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(c, d_C, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}

//verified
__global__ void GPU_matrix_transpose(double *m,double* n, int rows, int cols){

int i=blockIdx.x;
int j=threadIdx.x;

/*
if(j>i){
double aux=*(m+cols*i+j);
*(m+cols*i+j)=*(m+cols*j+i);
*(m+cols*j+i)=aux;
}
*/

if(i!=j){
*(n+cols*i+j)=*(m+cols*j+i);
}

}

//verified
double *matrix_transpose_CUDA(double *m, int rows, int cols){
    double *n=(double*)malloc(rows*cols*sizeof(double));

    cudaEvent_t start, stop;
    double *d_A, *d_B;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, rows*cols * sizeof(double)));
    //gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, m, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_B, B, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_transpose<<<blk_in_grid, thr_per_blk>>>(d_A,d_B,rows,cols);
    gpuErrchk(cudaEventRecord(stop));

    
    // Copy data from device array d_C to host array C
    cudaMemcpy(n, d_B, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    return n;


}

/*
__global__ void GPU_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){
    const int tpb=256;
    __shared__ double cache[tpb];
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    int cid=threadIdx.x;
    double sum=0;
    while(tid<)

}
*/

//verified
__global__ void GPU_matrix_mul(double* A,double* B,double* C,int a_rows, int a_cols, int b_rows, int b_cols){
    int i=blockIdx.x;
    int j=threadIdx.x;

    for(int n=0;n<a_cols;n++){
        *(C+i*blockDim.x+j) += *(A+i*blockDim.x+n) * *(B+j+n*blockDim.x);
    }

}

//verified
void matrix_mul_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){
    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int c_rows=a_rows;
    int c_cols=b_cols;

    gpuErrchk(cudaMalloc(&d_A, a_rows*a_cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, b_rows*b_cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, c_rows*c_cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, a, a_rows*a_cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, b, b_rows*b_cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(c_rows*c_cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_mul<<<blk_in_grid, thr_per_blk>>>(d_C,d_A,d_B,a_rows,a_cols,b_rows,b_cols);
    gpuErrchk(cudaEventRecord(stop));

   // Copy data from device array d_C to host array C
    cudaMemcpy(c, d_C, c_rows*c_cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

//void matrix_mul_trans_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){}



//verified
__global__ void GPU_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){

int j=blockIdx.x;
int i=threadIdx.x;
double sum=0;
int c_rows=a_rows;
int c_cols=b_cols;
int d_rows=c_rows;
int d_cols=c_cols;


if(j<c_rows && i<c_cols){
//printf("i=%i j=%i\n",i,j);
for(int n=0;n<a_cols;n++){
//sum += *(a+i*a_cols+n) * *(b+j+n*a_cols);
sum = sum + a[j*a_cols+n] * b[n*b_cols+i];
}
c[i*c_cols+j]=sum+d[i*d_cols+j];
//*(c+i*c_cols+j) = sum + *(d+i*d_cols+j);
}


}

//verified
void matrix_mul_add_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){
    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C, *d_D;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int c_cols=b_cols;
    int c_rows=a_rows;

    //gpuErrchk(cudaMalloc(&d_A, a_rows*a_cols * sizeof(double)));
    cudaMalloc(&d_A, a_rows*a_cols * sizeof(double));
    //gpuErrchk(cudaMalloc(&d_B, b_rows*b_cols * sizeof(double)));
    cudaMalloc(&d_B, b_rows*b_cols * sizeof(double));
    //gpuErrchk(cudaMalloc(&d_C, a_rows*b_cols * sizeof(double)));
    cudaMalloc(&d_C, a_rows*b_cols * sizeof(double));
    //gpuErrchk(cudaMalloc(&d_D, a_rows*b_cols * sizeof(double)));
    cudaMalloc(&d_D, a_rows*b_cols * sizeof(double));
    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, a, a_rows*a_cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, b, b_rows*b_cols* sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_D, d, a_rows*b_cols* sizeof(double), cudaMemcpyHostToDevice));
    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    thr_per_blk = c_cols;
    //blk_in_grid = ceil( (float)(c_rows*c_cols) / thr_per_blk );
    blk_in_grid = c_rows;
    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_mul_add<<<blk_in_grid, thr_per_blk>>>(d_C,d_A,d_B,a_rows,a_cols,b_rows,b_cols,d_D);
    gpuErrchk(cudaEventRecord(stop));

   // Copy data from device array d_C to host array C
    cudaMemcpy(c, d_C, c_rows*c_cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    //cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return;
}

__global__ void GPU_matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)){
int i=blockIdx.x;
int j=threadIdx.x;

if(i<rows && j<cols){
n[i*cols+j]=func(m[i*cols+j]);
}


}

void matrix_func_CUDA(double *n, double *m, int rows, int cols, double (*func)(double)){
cudaEvent_t start, stop;
    double *d_n, *d_m, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_n, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_m, rows*cols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, rows*cols * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_m, m, rows*cols* sizeof(double), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_B, B, rows*cols* sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)(rows*cols) / thr_per_blk );

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    GPU_matrix_func<<<blk_in_grid, thr_per_blk>>>(d_n,d_m,rows,cols,func);
    gpuErrchk(cudaEventRecord(stop));

    cudaMemcpy(n, d_n, rows*cols* sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_C);

}

