#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N (33*1024)


__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;  // handle the data at this index

    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    int a[N], b[N], c[N], i, M = 100;
    int *dev_a, *dev_b, *dev_c; 
  
    cudaMalloc((void**)&dev_c, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_a, N*sizeof(int));

    for(i=0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i*i*i;
    }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    // add <<<128, 128>>>(dev_a, dev_b, dev_c);
    // add <<<2, 2>>>(dev_a, dev_b, dev_c);
    add <<<(N+M-1)/M, M>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    for(i=0; i < N; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
    return 0;
}
