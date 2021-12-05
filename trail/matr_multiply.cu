#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thread block size
#define BLOCK_SIZE  16  // number of threads in a direction of the block
#define M_WIDTH     16 // number of columns
#define M_HEIGHT    16 // number of rows

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


using namespace std;
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    cout << "Copy d_C to C" << endl;
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}


int main(){
    float *A, *B, *C;
    int i, j;
    Matrix M_A, M_B, M_C; 

    C = (float*)malloc(M_WIDTH*M_HEIGHT*sizeof(float));  
    A = (float*)malloc(M_WIDTH*M_HEIGHT*sizeof(float));
    B = (float*)malloc(M_WIDTH*M_HEIGHT*sizeof(float));

    srand((unsigned)time( NULL ));

    // initialize A[] and B[]
    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            A[i*M_WIDTH + j] = (float)rand()/RAND_MAX;
            B[i*M_WIDTH + j] = (float)rand()/RAND_MAX;
        }
    }

    M_A.width = M_WIDTH; M_A.height = M_HEIGHT;
    M_A.elements = A; 
    M_B.width = M_WIDTH; M_B.height = M_HEIGHT;
    M_B.elements = B; 
    M_C.width = M_WIDTH; M_C.height = M_HEIGHT;
    M_C.elements = C; 

    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            cout <<A[i*M_WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << endl;
    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            cout <<B[i*M_WIDTH + j] << " ";
        }
        cout << endl;
    }

    cout << endl;
    cout << endl;

    MatMul(M_A, M_B,  M_C);

    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            cout <<M_A.elements[i*M_WIDTH + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    cout << endl;
    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            cout <<M_B.elements[i*M_WIDTH + j] << " ";
        }
        cout << endl;
    }

    cout << endl;
    cout << endl;

    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            cout <<M_C.elements[i*M_WIDTH + j] << " ";
        }
        cout << endl;
    }

    cout << endl;
    free(A); free(B); free(C);
    return 0;
}

