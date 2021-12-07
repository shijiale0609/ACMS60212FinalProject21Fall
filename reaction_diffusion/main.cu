#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N 200

__global__ void discrete_laplacian(double* M, double* L){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        L[tid] = (-4) * M[tid];
        if (tid%N!=0) {
            L[tid] += M[tid-1];
        } else {
            L[tid] += M[tid+N-1];
        }
        if (tid%N!=N-1) {
            L[tid] += M[tid+1];
        } else {
            L[tid] += M[tid-N+1];
        }
        if (tid/N!=0) {
            L[tid] += M[tid-N];
        } else {
            L[tid] += M[tid+N*N-N];
        }
        if (tid/N!=N-1) {
            L[tid] += M[tid+N];
        } else {
            L[tid] += M[tid-N*N+N];
        }
    }
}

__global__ void diff_Matrix_A(double* dev_A, double* dev_B, double* LA, double* diff_A, double DA, double f, double k, double delta_t){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        diff_A[tid] = (DA*LA[tid] - dev_A[tid]*dev_B[tid]*dev_B[tid] + f*(1-dev_A[tid])) * delta_t;
    }
}

__global__ void diff_Matrix_B(double* dev_A, double* dev_B, double* LB, double* diff_B, double DB, double f, double k, double delta_t){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        diff_B[tid] = (DB*LB[tid] + dev_A[tid]*dev_B[tid]*dev_B[tid] - (k+f)*dev_B[tid]) * delta_t;
    }
}

__global__ void add2to1(double* M1, double* M2){
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N*N) {
        M1[tid] += M2[tid];
    }
}

void gray_scott_update(double* dev_A, double* dev_B, double* LA, double* LB, double* diff_A, double* diff_B, double DA, double DB, double f, double k, double delta_t){
    discrete_laplacian<<<N*N,1>>>(dev_A, LA);
    discrete_laplacian<<<N*N,1>>>(dev_B, LB);
    diff_Matrix_A<<<N*N,1>>>(dev_A,dev_B,LA,diff_A,DA,f,k,delta_t);
    diff_Matrix_B<<<N*N,1>>>(dev_A,dev_B,LB,diff_B,DB,f,k,delta_t);
    add2to1<<<N*N,1>>>(dev_A,diff_A);
    add2to1<<<N*N,1>>>(dev_B,diff_B);
}

void get_initial_configuration(double* A0, double* B0, double random_influence){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int N2 = N/2, r = N/10;
            if (i < N2+r && i >= N2-r && j < N2+r && j >= N2-r) {
                A0[N*i+j] = 0.50;
                B0[N*i+j] = 0.25;
            } else {
                A0[N*i+j] = 1 - random_influence + random_influence * ((double) rand() / (RAND_MAX));
                B0[N*i+j] = random_influence * ((double) rand() / (RAND_MAX));
            }
        }
    }
}

void output_txt(double* A0, double* B0, double* A, double* B){
    FILE *fptr;
    fptr = fopen("A0.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", A0[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
    fptr = fopen("B0.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", B0[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
    fptr = fopen("A.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", A[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
    fptr = fopen("B.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", B[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
}

int main () {
    double delta_t = 1.0;
    double DA = 0.16; // second set: 0.14
    double DB = 0.08; // second set: 0.06
    double f = 0.060; // second set: 0.035
    double k = 0.062; // second set: 0.065
    int N_simulation_steps = 10000;
    double random_influence = 0.2;
    double *dev_A, *dev_B, *LA, *LB, *diff_A, *diff_B;
    double A[N*N], B[N*N], A0[N*N], B0[N*N];
    cudaMalloc((void**)&dev_A, N*N*sizeof(double));
    cudaMalloc((void**)&dev_B, N*N*sizeof(double));
    cudaMalloc((void**)&LA, N*N*sizeof(double));
    cudaMalloc((void**)&LB, N*N*sizeof(double));
    cudaMalloc((void**)&diff_A, N*N*sizeof(double));
    cudaMalloc((void**)&diff_B, N*N*sizeof(double));
    get_initial_configuration(A0, B0, random_influence);
    cudaMemcpy(dev_A, A0, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B0, N*N*sizeof(double), cudaMemcpyHostToDevice);
    for (int t=0; t<N_simulation_steps; t++) {
        gray_scott_update(dev_A, dev_B, LA, LB, diff_A, diff_B, DA, DB, f, k, delta_t);
    }
    cudaMemcpy(A, dev_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, dev_B, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    output_txt(A0, B0, A, B);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(LA);
    cudaFree(LB);
    cudaFree(diff_A);
    cudaFree(diff_B);
    return 0;
}
