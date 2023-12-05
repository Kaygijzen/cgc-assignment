#ifndef _MODULE_H_
#define _MODULE_H_

#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

#include <numeric>
#include <cuda.h>

// CUDA_GLOBAL void add_vectors(double *a, double *b, double *c);
void call_kernel(int a);

#endif