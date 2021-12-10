#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>

#include "Gray_Scott.h"


int get_initial_configuration(float *A, float *B, int N, float random_influence);
int write_output(float *M, char* name , int N);

using namespace std;

int main(){
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

  // random influence factor
  float random_influence = 0.2;

  float A0[N*N], B0[N*N], A1[N*N], B1[N*N];

  // initialization
  get_initial_configuration(A0, B0, N, random_influence);

  // update the matrices
  gray_scott_update(A0, B0, A1, B1, DA, DB, f, k, delta_t, N, N_simulation_steps);

  // save the results
  write_output(A0,"A0.txt",N);
  write_output(B0,"B0.txt",N);
  write_output(A1,"A1.txt",N);
  write_output(B1,"B1.txt",N);

}


int get_initial_configuration(float *A, float *B, int N, float random_influence){
  int i,j;
  int N2 = N/2, r = N/10;

  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      if (i < N2+r && i >= N2-r && j < N2+r && j >= N2-r) {
	A[N*i+j] = 0.50;
	B[N*i+j] = 0.25;
      } else {
	A[N*i+j] = 1 - random_influence + random_influence * ((float) rand() / (RAND_MAX));
	B[N*i+j] = random_influence * ((float) rand() / (RAND_MAX));
      }
    }
  }
  
  return 0;
}

int write_output(float *M, char* name , int N){
  FILE *fptr;
  fptr = fopen(name,"w");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fprintf(fptr, "%lf\t", M[N*i+j]);
    }
    fprintf(fptr, "\n");
  }

}
