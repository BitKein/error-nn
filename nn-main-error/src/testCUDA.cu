#include "nn.h"
#include "../include/matrixCUDA.cuh"
#include "matrix.h"
#include "../include/testCUDA.cuh"

void forward_pass_test_CUDA(nn_t *nn, double *input, double **A){
    int i;

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){
/*	if(i==1){
	printf("Primera fila de A: %f %f %f\n", *((nn->WH[i - 1])+0),*((nn->WH[i - 1])+1),*((nn->WH[i - 1])+2));
	printf("Primera fila de B: %f %f %f\n", *(A[i - 1]+0),*(A[i - 1]+1),*(A[i - 1]+2));
	printf("Primera fila de D: %f %f %f\n", *((nn->BH[i - 1])+0),*((nn->BH[i - 1])+1),*((nn->BH[i - 1])+2));

	}
*/
        matrix_mul_add_CUDA(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func_CUDA(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }
}


