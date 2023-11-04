import numpy as np
import os
from string import Template
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import glob

#filename = sys.argv[1]

def read_data(infile):
    time = []
    energy = []
    value = []
    with open(infile) as read_file:
        lines = read_file.readlines()
        for i in lines:
            time.append(i.split()[0])
            energy.append(i.split()[1])
            value.append(i.split()[2])
    return time, energy, value
    
def plot_signal(energy, value, outname='spectrum.png'):
    value = np.reshape(value, (int(len(value)/len(energy)),len(energy))).astype(np.float32)
    value = np.nan_to_num(value) 
    energy = np.array(energy, dtype=float) 
    fig, ax = plt.subplots(1,1)
    c = ax.pcolormesh(energy, energy, value, cmap='rainbow', vmin=-0.2, vmax=0.8, shading='nearest')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r'$\hbar\omega_{\tau}$, [eV]', fontsize=20)
    ax.set_ylabel(r'$\hbar\omega_{t}$, [eV]', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    plt.show()
   # plt.savefig(outname, bbox_inches='tight',dpi=400)

