import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.figsize"] = (8,4)

def make_table(files):
    counter = 0
    table = np.zeros((len(files),2))
    for file in files:
        Z = np.loadtxt(file,delimiter=',')
        M = Z.shape[0] - 1
        Z_outcome = 1 if Z[-1]>5000 else 0
        table[counter,:] = np.array([Z_outcome, M])
        counter+=1
    return table

folder = 'data/land-dist/'
files = glob.glob(f'{folder}*-Z.out')

table = make_table(files)

np.savetxt('sweep.out',table)