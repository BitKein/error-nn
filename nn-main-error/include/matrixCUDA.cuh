
#ifndef __MATRIXCUDA_H
#define __MATRIXCUDA_H

#include <stdbool.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define gpuErrchk(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


void matrix_sum_CUDA(double *c, double *a, double *b, int rows, int cols);

void matrix_sub_CUDA(double *c, double *a, double *b, int rows, int cols);

void matrix_mul_cnt_CUDA(double *m, int rows, int cols, double cnt);

void matrix_zero_CUDA(double *m, int rows, int cols);

void matrix_mul_dot_CUDA(double *c, double *a, double *b, int rows, int cols);

double *matrix_transpose_CUDA(double *m, int rows, int cols);

void matrix_mul_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void matrix_mul_trans_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

void matrix_mul_add_CUDA(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d);

void matrix_func_CUDA(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

void print_matrix_CUDA(double *m, int m_rows, int m_cols);

#endif
