#include <iostream>
#include <new>
#include <cstddef>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vec_arb_len_add.h"
#include "vec_arb_len_add_constant.h"
#include "vec_arb_len_dot.h"
#include "vec_arb_len_scale.h"
#include "vec_arb_len_copy.h"

using namespace std;


int save_write(float*, float*, int, string);

int discrete_laplacian(float*, float*, int);

int get_initial_configuration(float*, float*, int, float);

int gray_scott_update(float*, float*, float,float, float, float, float, int);

int main(int argc, char *argv[])
{

        // update in time
        float delta_t = 1.0;
        
        // Diffusion coefficients
        float DA = 0.14;
        float DB = 0.06;

        // define feed/kill rates
        float f = 0.035;
        float k = 0.065;

        // grid size
        int N = 200;
        
        // simulation steps
        int N_simulation_steps = 10000;
        float random_influence = 0.2;

        float  *A, *B;
        
        //float *TEMP;        
        
        A = (float *)malloc(N * N * sizeof(float));
        B = (float *)malloc(N * N * sizeof(float));
        //TEMP = (float *)malloc(N * N * sizeof(float));
        

       int i,j; 

   
        // initialize A and B
        get_initial_configuration(A, B, N, random_influence);
       
        save_write(A,B,N, "Input");
        // update A and B
        for (i=0; i<N_simulation_steps;i++)
        {

               gray_scott_update(A,B, DA, DB, f, k, delta_t, N);    
        }
        
        cout << "Save the output" << endl;
        save_write(A,B,N, "Output");
        free(A);
        free(B);
      //free(TEMP);

      return 0;
}

int save_write(float *A, float *B , int N, string Str)
{
   

    // open a file in write mode.
    std::ofstream outfileA;
    string stringA = Str+"A";
    outfileA.open(stringA);

    int i,j;
    //outfile << "Solution vector:" << std::endl;
    for(i = 0; i <N; i++)
    {
        for (j = 0; j<N; j++)
        {
           outfileA << A[i*N+j] <<" ";
        }
        outfileA<<std::endl;
    }    
        outfileA<<std::endl;

   // close the opened file.
   outfileA.close();

    // open a file in write mode.
    std::ofstream outfileB;
    string stringB = Str+"B";
    outfileB.open(stringB);

    //outfile << "Solution vector:" << std::endl;
    for(i = 0; i <N; i++)
    {
        for (j = 0; j<N; j++)
        {
           outfileB << B[i*N+j] <<" ";
        }
        outfileB<<std::endl;
    }    
        //outfileB<<std::endl;

   // close the opened file.
   outfileB.close();



    return 0;
}


int discrete_laplacian(float* Matrix, float* Matrix_laplacian, int N)
{
  
   float  *Left, *Right, *Bottom, *Top;
   float  *MinusFourMatrix;
   float  *Temp1, *Temp2;   

   Left =   (float *)malloc(N * N * sizeof(float));
   Right =  (float *)malloc(N * N * sizeof(float));
   Bottom = (float *)malloc(N * N * sizeof(float));
   Top =    (float *)malloc(N * N * sizeof(float));
   
   int i,j;

   // left
   for(i=0;i<N;i++)
   {
      for(j=0;j<N;j++)
      {
        Left[i*N+j] = Matrix[i*N+(j+N-1)%N];     
      }
   }

   // right
   for(i=0;i<N;i++)
   {
      for(j=0;j<N;j++)
      {
        Right[i*N+j] = Matrix[i*N+(j+1)%N];     
      }
   }

   // bottom
   for(i=0;i<N;i++)
   {
      for(j=0;j<N;j++)
      {
        Bottom[i*N+j] = Matrix[(i*N+j+N*(N-1))%(N*N)];     
      }
   }

   // top
   for(i=0;i<N;i++)
   {
      for(j=0;j<N;j++)
      {
        Top[i*N+j] = Matrix[(i*N+j+N)%(N*N)];     
      }
   }


   MinusFourMatrix =  (float *)malloc(N * N * sizeof(float));
   cuda_vec_scale(Matrix, -4, MinusFourMatrix, N);

   Temp1 =  (float *)malloc(N * N * sizeof(float));
   Temp2 =  (float *)malloc(N * N * sizeof(float));
   
   cuda_vec_add(Left, Right, Temp1, N);
   cuda_vec_add(Temp1, Bottom, Temp2, N);
   cuda_vec_add(Temp2, Top, Temp1, N);
   cuda_vec_add(Temp1, MinusFourMatrix, Temp2, N);

   cuda_vec_copy(Temp2, Matrix_laplacian, N ); 
   
   free(Left);
   free(Right);
   free(Bottom);
   free(Top);
   free(MinusFourMatrix);
   free(Temp1);
   free(Temp2);

   return 0;
}

int get_initial_configuration(float *A, float *B, int N, float random_influence)
{
    cout << "Initialize A and B" << endl;
    int i,j;

    int N2 = N/2;
    int r = N/10;

    for (i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            if( i > N2-r and i < N2+r and j > N2-r and j < N2+r)
            {

               A[i*N+j] = 0.50;
               B[i*N+j] = 0.25;

            }
            else
            {
    
               A[i*N+j] = (1.0 - random_influence) + random_influence * (float)rand()/RAND_MAX;
               B[i*N+j] = random_influence * (float)rand()/RAND_MAX;

            } 
        }
    }
     
  return 0;
}

int gray_scott_update(float *A, float *B, float DA, float DB, float f, float k, float delta_t, int N)
{
   
   float  *LA, *LB;
   float  *diff_A, *diff_B;
   
   float  *Temp1, *Temp2, *Temp3;
   float  *TempA, *TempB;

   float  *TempA1, *TempB1;
   float  *TempA2, *TempB2;
   float *minusfA, *fminusfA, *minuskfB;

   LA =   (float *)malloc(N * N * sizeof(float));
   LB =   (float *)malloc(N * N * sizeof(float));
   diff_A =   (float *)malloc(N * N * sizeof(float));
   diff_B =   (float *)malloc(N * N * sizeof(float));
   TempA =   (float *)malloc(N * N * sizeof(float));
   TempA1 =   (float *)malloc(N * N * sizeof(float));
   TempA2 =   (float *)malloc(N * N * sizeof(float));

   TempB =   (float *)malloc(N * N * sizeof(float));
   TempB1 =   (float *)malloc(N * N * sizeof(float));
   TempB2 =   (float *)malloc(N * N * sizeof(float));

   Temp1 =   (float *)malloc(N * N * sizeof(float));
   Temp2 =   (float *)malloc(N * N * sizeof(float));
   Temp3 =   (float *)malloc(N * N * sizeof(float));
   
   minusfA = (float *)malloc(N * N * sizeof(float));
   fminusfA = (float *)malloc(N * N * sizeof(float));
   minuskfB = (float *)malloc(N * N * sizeof(float));

 
   discrete_laplacian(A, LA, N);
   discrete_laplacian(B, LB, N);
   

   cuda_vec_scale(LA, DA, TempA1, N);
   cuda_vec_scale(LB, DB, TempB1, N);

   // Temp1 = B*B
   cuda_vec_dot(B,B, Temp1, N);
   // Temp2 = A*B*B
   cuda_vec_dot(A,Temp1, Temp2, N);
   
   // Temp3 = -A*B*B = Temp2*(-1)
   cuda_vec_scale(Temp2, -1.0, Temp3, N);

   //
   cuda_vec_scale(A, -1.0*f, minusfA, N);
   cuda_vec_add_constant(minusfA, f, fminusfA, N);

   cuda_vec_add(TempA1, Temp3, TempA, N);
   cuda_vec_add(TempA, fminusfA, TempA2, N);
   cuda_vec_scale(TempA2, delta_t, diff_A, N);
   
   cuda_vec_add(diff_A, A, TempA, N);
   cuda_vec_copy(TempA, A, N);
    

   cuda_vec_scale(B,-1.0*(k+f), minuskfB, N);

   cuda_vec_add(TempB1, Temp2, TempB, N);
   cuda_vec_add(TempB, minuskfB, TempB2, N);
   cuda_vec_scale(TempB2, delta_t, diff_B, N);
   
   cuda_vec_add(diff_B, B, TempB, N);
   cuda_vec_copy(TempB, B, N); 
     

   free(LA);
   free(LB);

   free(diff_A);
   free(diff_B);
   free(TempA);
   free(TempA1); 
   free(TempA2); 

   free(TempB); 
   free(TempB1);
   free(TempB2);

   free(Temp1); 
   free(Temp2);
   free(Temp3);
   
   free(minusfA); 
   free(fminusfA);
   free(minuskfB);

   return 0;
}
