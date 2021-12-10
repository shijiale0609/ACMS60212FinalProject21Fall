import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def draw(A,B,name):
    """draw the concentrations"""
    fig, ax = plt.subplots(1,2,figsize=(5.65,4))
    ax[0].imshow(A, cmap='Greys')
    ax[1].imshow(B, cmap='Greys')
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(name)

N = 200
A0,B0,A1,B1= np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
A0 = pd.read_csv('A0.txt', sep="\t", header=None)
B0 = pd.read_csv('B0.txt', sep="\t", header=None)
A = pd.read_csv('A1.txt', sep="\t", header=None)
B = pd.read_csv('B1.txt', sep="\t", header=None)
draw(A0,B0, "Initial_State.png")
draw(A,B, "Final_State.png")
