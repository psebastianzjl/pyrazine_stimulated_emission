import numpy as np
import os
from string import Template
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import glob


def read_data(infile):
    time = []
    energy = []
    value = []
    with open(infile) as read_file:
        lines = read_file.readlines()
        for i in lines:
            time.append(float(i.split()[0]))
            energy.append(float(i.split()[1]))
            value.append(float(i.split()[2]))
    return time, energy, value
    
def plot_signal(time, energy, value, outname='spectrum.png'):
    value = np.reshape(value, (int(len(value)/401),401)).astype(np.float32)
    value = np.nan_to_num(value) 
    time = np.array(time, dtype=float)
    time_new =  time
    energy = np.array(energy, dtype=float) 
    fig, ax = plt.subplots(1,1)
    c = ax.pcolormesh(time_new, energy, value, cmap='rainbow', vmin=0.0, vmax=0.1, shading='nearest')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14) 
    ax.axis([time_new.min(), time_new.max(), energy.min(), energy.max()])

    ax.set_xlabel(r'population time $T$, [fs]', fontsize=20)
    ax.set_ylabel(r'$\hbar\omega_{pr}$, [eV]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    plt.show()
   # plt.savefig(outname, bbox_inches='tight',dpi=400)

