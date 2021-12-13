#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N 200 // N is the side length of the (N x N)-sized grid. stays constant throughout the program

__global__ void discrete_laplacian(float* M, float* L){
    //get the discrete Laplacian of matrix M
    // do so by summing over neighboring cells and subtracting 4*value of original cell
    // values at each index stored in L, update L 
    
    int tid = blockIdx.x; //handle data at this index
    if (tid < N*N){ //make sure tid is valid
        L[tid] = (-4) * M[tid];

        if (tid%N != N-1){ //add right neighbor
            L[tid] += M[tid + 1];
        } else {
            L[tid] += M[tid - N + 1];
        }

        if (tid%N != 0){ //add left neighbor
            L[tid] += M[tid - 1];
        } else {
            L[tid] += M[tid + N - 1];
        }

        if (tid/N != 0){ //add top neighbor
            L[tid] += M[tid - N];
        } else {
            L[tid] += M[tid + N*N - N];
        }

        if (tid%N != N-1){ //add bottom neighbor
            L[tid] += M[tid + N];
        } else {
            L[tid] += M[tid- N*N + N];
        }
    }   
}


__global__ void get_diff_A(float* A, float* B, float* LA, float* diff_A, float DA, float f, float delta_t){
    // update formula for chemical A
    int tid = blockIdx.x;
    if (tid < N*N) {
        diff_A[tid] = (DA*LA[tid] - A[tid]*B[tid]*B[tid] + f*(1-A[tid])) * delta_t;
    }
}

__global__ void get_diff_B(float* A, float* B, float* LB, float* diff_B, float DB, float f, float k, float delta_t){
    // update formula for chemical B
    int tid = blockIdx.x;
    if (tid < N*N) {
        diff_B[tid] = (DB*LB[tid] + A[tid]*B[tid]*B[tid] - (k+f)*B[tid]) * delta_t;
    }
}

__global__ void mat_add(float* M1, float* M2){
    // this function is used to update matrix A (or B) by adding diff_A (or diff_B)
    int tid = blockIdx.x;
    if (tid < N*N) {
        M1[tid] += M2[tid];
    }
}

void gray_scott_update(float* A, float* B, float* LA, float* LB, float* diff_A, float* diff_B, float DA, float DB, float f, float k, float delta_t){
    discrete_laplacian<<<N*N,1>>>(A, LA);
    discrete_laplacian<<<N*N,1>>>(B, LB);
    get_diff_A<<<N*N,1>>>(A, B, LA, diff_A, DA, f, delta_t);
    get_diff_B<<<N*N,1>>>(A, B, LB, diff_B, DB, f, k, delta_t);
    mat_add<<<N*N,1>>>(A,diff_A);
    mat_add<<<N*N,1>>>(B,diff_B);
}

void get_initial_configuration(float* A0, float* B0, float random_influence){
    // create matrices A0 and B0 to represent initial concentrations of chemicals A and B
    // random_influence describes how much noise is added
    
    int N2 = N/2, r = N/10; // setting radius of disturbance from the center
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i < N2-r || i >= N2+r || j < N2-r || j >= N2+r) {
                A0[N*i+j] = 1 - random_influence + random_influence * ((float) rand() / (RAND_MAX)); // initial concentration of A at every cell is (1-random_influence) * 1 + random_influence * random_number
                B0[N*i+j] = random_influence * ((float) rand() / (RAND_MAX)); // initial concentration of B at every cell is random_influence * random_number
            } else { // adding disturbance in the center
                A0[N*i+j] = 0.50;
                B0[N*i+j] = 0.25;        
            }
        }
    }
}

void output_txt(float* A0, float* B0, float* A1, float* B1){
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
    fptr = fopen("A1.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", A1[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
    fptr = fopen("B1.txt","w");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%lf\t", B1[N*i+j]);
        }
        fprintf(fptr, "\n");
    }
}

int main () {
    // define paratemers
    float delta_t = 1.0;    // update in time
    float DA = 0.16;  // diffusion coefficients of A
    float DB = 0.08; // diffusion coefficients of B
    float f = 0.060; // define feed rate
    float k = 0.062;  // define kill rate
    int N_simulation_steps = 10000;
    float random_influence = 0.2;
    float A0[N*N], B0[N*N], A1[N*N], B1[N*N];
    float *A, *B, *LA, *LB, *diff_A, *diff_B;

    // allocate memory
    cudaMalloc((void**)&A, N*N*sizeof(float));
    cudaMalloc((void**)&B, N*N*sizeof(float));
    cudaMalloc((void**)&LA, N*N*sizeof(float));
    cudaMalloc((void**)&LB, N*N*sizeof(float));
    cudaMalloc((void**)&diff_A, N*N*sizeof(float));
    cudaMalloc((void**)&diff_B, N*N*sizeof(float));

    //initialize and save A0, B0 into A, B
    get_initial_configuration(A0, B0, random_influence);
    cudaMemcpy(A, A0, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B0, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // update A, B
    for (int t=0; t<N_simulation_steps; t++) {
        gray_scott_update(A, B, LA, LB, diff_A, diff_B, DA, DB, f, k, delta_t);
    }

    // save the updated A, B into final state A1, B1
    cudaMemcpy(A1, A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B1, B, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // output into txt for plot
    output_txt(A0, B0, A1, B1);

    // free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(LA);
    cudaFree(LB);
    cudaFree(diff_A);
    cudaFree(diff_B);
    return 0;
}
