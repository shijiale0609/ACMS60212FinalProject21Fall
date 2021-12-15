#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vec_arb_len_add_constant.h"

#define BLOCK_SIZE  16  // number of threads in a direction of the block

using namespace std;

__global__ void add_constant(float *a, float lambda, float *c, int N){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;  // handle the data at this index

    while(tid < N*N)
    {
        c[tid] = a[tid] + lambda;
        tid += blockDim.x * gridDim.x;
    }
}

int cuda_vec_add_constant( float *a, float lambda, float *c, int N)
{
    //int N = 9;
    //int i,j;

    //float *a, *b, *c;
    //c = (float*)malloc(N*N*sizeof(float));  
    //a = (float*)malloc(N*N*sizeof(float));
    //b = (float*)malloc(N*N*sizeof(float));

    float *dev_a,  *dev_c; 
  
    cudaMalloc((void**)&dev_c, N*N*sizeof(float));
    //cudaMalloc((void**)&dev_b, N*N*sizeof(float));
    cudaMalloc((void**)&dev_a, N*N*sizeof(float));

    cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

    add_constant <<<(N*N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dev_a, lambda, dev_c, N);

    cudaMemcpy(c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);
 
    cudaFree(dev_c);
    //cudaFree(dev_b);
    cudaFree(dev_a);


    return 0;
}
