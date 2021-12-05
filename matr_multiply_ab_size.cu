#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thread block size
#define BLOCK_SIZE  16  // number of threads in a direction of the block
#define M_WIDTH     522 // number of columns. 512
#define M_HEIGHT    522 // number of rows

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    double* elements;
} Matrix;


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(double);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width+BLOCK_SIZE-1) / dimBlock.x, (A.height+BLOCK_SIZE-1)/ dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

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
    double Cvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M_HEIGHT && col < M_WIDTH)
    {
        for (int e = 0; e < A.width; ++e)
            Cvalue += A.elements[row * A.width + e]
                    * B.elements[e * B.width + col];
        C.elements[row * C.width + col] = Cvalue;
    }
}


int main(){
    double *A, *B, *C, *cpu_C;
    int i, j;
    Matrix M_A, M_B, M_C; 

    cpu_C = (double*)malloc(M_WIDTH*M_HEIGHT*sizeof(double));  
    C = (double*)malloc(M_WIDTH*M_HEIGHT*sizeof(double));  
    A = (double*)malloc(M_WIDTH*M_HEIGHT*sizeof(double));
    B = (double*)malloc(M_WIDTH*M_HEIGHT*sizeof(double));

    srand((unsigned)time( NULL ));

    // initialize A[] and B[]
    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            A[i*M_WIDTH + j] = (double)rand()/RAND_MAX;
            B[i*M_WIDTH + j] = (double)rand()/RAND_MAX;
            cpu_C[i*M_WIDTH + j] = 0.0;
        }
    }


    M_A.width = M_WIDTH; M_A.height = M_HEIGHT;
    M_A.elements = A; 
    M_B.width = M_WIDTH; M_B.height = M_HEIGHT;
    M_B.elements = B; 
    M_C.width = M_WIDTH; M_C.height = M_HEIGHT;
    M_C.elements = C; 

    double Cvalue; 
    for(i = 0; i < M_HEIGHT; i++)
    {
        for(j = 0; j < M_WIDTH; j++)
        {
            Cvalue = 0.0; 
            for (int e = 0; e < M_A.width; ++e)
                Cvalue += M_A.elements[i * M_A.width + e]
                          *M_B.elements[e * M_B.width + j];
            cpu_C[i*M_WIDTH + j] = Cvalue;
        }
    }

    MatMul(M_A, M_B,  M_C);
    for(i = 0; i < M_WIDTH; i++)
    {
        for(j = 0; j < 1; j++)
        {
            printf("difference[%d] cpu-gpu = %g\n", 
                  i, fabs(cpu_C[i*M_WIDTH + j] - 
                M_C.elements[i*M_C.width + j]) );
        }
    }

    free(A); free(B); free(C); free(cpu_C); 
    return 0;
}

