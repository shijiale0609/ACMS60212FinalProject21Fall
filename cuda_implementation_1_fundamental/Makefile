# This is a comment line 
CUDA_HOME :=/afs/crc.nd.edu/x86_64_linux/c/cuda/11.0

CC=icpc 
NVCC = nvcc
# CFLAGS will be the options passed to the compiler. 
CFLAGS= -c -Wall -std=c++11
OBJECTS  = main.o  vec_arb_len_add.o  vec_arb_len_dot.o vec_arb_len_scale.o vec_arb_len_copy.o vec_arb_len_add_constant.o
DEBUG_FLAG = -g
C_LIBS = -ljpeg

# NB! arch=sm_35 is the lowest supported for dynamic parallelism
NVCCINC = -I $(CUDA_HOME)/include -I.
NVCCFLAGS = -lineinfo -arch=sm_50 
NVCCLIB     = -L $(CUDA_HOME)/lib64 -lcudadevrt -lcublas_static -lculibos -lcudart

all: prog 

prog:  $(OBJECTS)
	$(CC)  $(OBJECTS)  $(DEBUG_FLAG) $(C_LIBS) $(NVCCLIB) -o $@ 

%.o: %.cpp 
	$(CC) $(CFLAGS) $(DEBUG_FLAG) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINC) -o $@ -c $<
clean: 
	rm -rf   *.o 

tagsfile:
	ctags *.cpp *.h *.cu
