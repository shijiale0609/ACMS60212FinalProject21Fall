#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "Gray_Scott.h"

// for the sake of simplicity, use 1-D structure

// Discrete Laplacian
__global__ void discrete_laplacian(float* M, float* L, int N){
    int tid = blockIdx.x; // handle the data at this index

    if (tid < N*N) {
        L[tid] = (-4) * M[tid];

        if (tid%N!=0) {		//left
            L[tid] += M[tid-1];
        } else {
            L[tid] += M[tid+N-1];
        }
        if (tid%N!=N-1) {	//right
            L[tid] += M[tid+1];
        } else {
            L[tid] += M[tid-N+1];
        }
        if (tid/N!=0) {		//top
            L[tid] += M[tid-N];
        } else {
            L[tid] += M[tid+N*N-N];
        }
        if (tid/N!=N-1) {	//bottom
            L[tid] += M[tid+N];
        } else {
            L[tid] += M[tid-N*N+N];
        }
    }
}


// update formula
__global__ void diff_A_funx(float* A, float* B, float* LA, float* diff_A, float DA, float f, float k, float delta_t,int N){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        diff_A[tid] = (DA*LA[tid] - A[tid]*B[tid]*B[tid] + f*(1-A[tid])) * delta_t;
    }
}

__global__ void diff_B_funx(float* A, float* B, float* LB, float* diff_B, float DB, float f, float k, float delta_t,int N){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
    diff_B[tid] = (DB*LB[tid] + A[tid]*B[tid]*B[tid] - (k+f)*B[tid]) * delta_t;
    }
}

__global__ void add(float* M1, float* M2, int N){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        M1[tid] += M2[tid];
    }
}


// update function
void gray_scott_update(float* A0, float* B0, float* A1, float* B1, float DA, float DB, float f, float k, float delta_t, int N, int N_simulation_steps){

    float *A, *B, *LA, *LB, *diff_A, *diff_B;
    int t;

    cudaMalloc((void**)&LA, N*N*sizeof(float));
    cudaMalloc((void**)&LB, N*N*sizeof(float));
    cudaMalloc((void**)&diff_A, N*N*sizeof(float));
    cudaMalloc((void**)&diff_B, N*N*sizeof(float));
    cudaMalloc((void**)&A, N*N*sizeof(float));
    cudaMalloc((void**)&B, N*N*sizeof(float));
    
    cudaMemcpy(A, A0, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B0, N*N*sizeof(float), cudaMemcpyHostToDevice);

    for (t=0; t<N_simulation_steps; t++){
    	 discrete_laplacian<<<N*N,1>>>(A, LA, N);
    	 discrete_laplacian<<<N*N,1>>>(B, LB, N);
    	 diff_A_funx<<<N*N,1>>>(A,B,LA,diff_A,DA,f,k,delta_t,N);
    	 diff_B_funx<<<N*N,1>>>(A,B,LB,diff_B,DB,f,k,delta_t,N);
    	 add<<<N*N,1>>>(A,diff_A,N);
    	 add<<<N*N,1>>>(B,diff_B,N);
    }
    
    cudaMemcpy(A1, A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B1, B, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(LA);
    cudaFree(LB);
    cudaFree(diff_A);
    cudaFree(diff_B);

}

