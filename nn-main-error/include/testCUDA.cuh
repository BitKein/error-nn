#ifndef __TESTCUDA_H
#define __TESTCUDA_H

void forward_pass_test_CUDA(nn_t *nn, double *input, double **A);

float precision(int tp, int fp);

float recall(int tp, int fn);

float f1(float p, float r);

#endif

