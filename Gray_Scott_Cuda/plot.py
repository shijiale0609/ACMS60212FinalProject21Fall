import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def draw(A,B, Str):
    """draw the concentrations"""
    fig, ax = plt.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig("Gray_Scott_" + Str +".png")


N = 200
A = np.ones((N,N))
B = np.ones((N,N))

i = 0
with open("InputA") as f:
     for line in f:
           if i < N :
               cols = line.split()
               #print(i,len(cols))
               for j in range(0, N):
                  #print(i, cols[j])
                  A[i,j] = float(cols[j])
               i = i+1


i = 0
with open("InputB") as f:
     for line in f:
           if i < N :
               cols = line.split()
               #print(i,len(cols))
               for j in range(0, N):
                  #print(i, cols[j])
                  B[i,j] = float(cols[j])
               i = i+1




draw(A,B, "Input")



i = 0
with open("OutputA") as f:
     for line in f:
           if i < N :
               cols = line.split()
               #print(i,len(cols))
               for j in range(0, N):
                  #print(i, cols[j])
                  A[i,j] = float(cols[j])
               i = i+1


i = 0
with open("OutputB") as f:
     for line in f:
           if i < N :
               cols = line.split()
               #print(i,len(cols))
               for j in range(0, N):
                  #print(i, cols[j])
                  B[i,j] = float(cols[j])
               i = i+1




draw(A,B, "Output")



