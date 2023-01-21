#ifndef __TRAINCUDA_H
#define __TRAINCUDA_H

#include "nn.h"
#include "nn_aux.h"
#include "ds.h"
#include "matrix.h"
#include "matrixCUDA.cuh"

void forward_pass_CUDA(nn_t *nn, double *input, double **A, double **Z);

double back_prop_CUDA(nn_t *nn, double *output, double **A, double **Z, double **D, double **d);

void update_CUDA(nn_t *nn, double **D, double **d, double lr, int batch_size);

#endif
