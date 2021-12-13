#this part is taken from the colab notebook
#it contains a plot function which is used to draw initial configureations and final concentrations


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#parameters are concentrations of chemical A and B, and output file name
def draw(A,B,output_file_name):
    """draw the concentrations"""
    fig, ax = plt.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(output_file_name)


#the rest of the code uses c++ and the plotting part uses python
#so the input is read from txt files generated by the c++ part of the code
N = 200
A0, B0 ,A1, B1= np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
A0 = pd.read_csv('A0.txt', sep="\t", header=None)
B0 = pd.read_csv('B0.txt', sep="\t", header=None)
A1 = pd.read_csv('A1.txt', sep="\t", header=None)
B1 = pd.read_csv('B1.txt', sep="\t", header=None)
draw(A0,B0, "Initial_State.png")
draw(A1,B1, "Final_State.png")
