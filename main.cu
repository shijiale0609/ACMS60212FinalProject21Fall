#include <iostream>
#include <new>
#include <cstddef>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


using namespace std;


int discrete_laplacian(float*);

int get_initial_configuration(float*, float*, int, float);

int gray_scott_update(float*, float*, float,float, float, float, float );

int main(int argc, char *argv[])
{

        // update in time
        float delta_t = 1.0;
        
        // Diffusion coefficients
        float DA = 0.16;
        float DB = 0.08;

        // define feed/kill rates
        float f = 0.060;
        float k = 0.062;

        // grid size
        int N = 200;
        
        // simulation steps
        int N_simulation_steps = 10000;
        float random_influence = 0.2;

        float  *A, *B;
        
        
        A = (float *)malloc(N * N * sizeof(float));
        B = (float *)malloc(N * N * sizeof(float));
        

        // initialize A and B
        get_initial_configuration(A, B, N, random_influence);
       
        int i;
        // update A and B
        for (i=0; i<N_simulation_steps;i++)
        {

               gray_scott_update(A,B, DA, DB, f, k, delta_t);    
        }
}

int discrete_laplacian(float* M)
{
   return 0;
}

int get_initial_configuration(float *A, float *B, int N, float random_influence)
{
  cout << "Initialize A and B" << endl;   
  return 0;
}

int gray_scott_update(float *A, float *B, float DA, float DB, float f, float k, float delta_t )
{
   return 0;
}
