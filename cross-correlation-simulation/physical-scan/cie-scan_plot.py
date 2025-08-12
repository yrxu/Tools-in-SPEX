#!/usr/bin/env python
#run source $SPEX90/spexdist.sh first!
import numpy as np
import pandas as pd
import os
import time
import subprocess
import multiprocessing as mp
import shutil
from glob import glob
import itertools
from pyspex import Session
import scipy.stats as stats
import xarray as xr
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import combinations


### provide the name of the global sigma map and parameters
ID="202403"
work_dir='CC-MC-cie_'+ID
out_global_sigma   = work_dir+"/global_sigma_map.npy"
global_sigma_map = np.load(out_global_sigma)
sys_zvs = np.arange(-45000,45000,step=1000)
temperatures = np.logspace(np.log10(0.2),np.log10(4),90)

zv_vals = sys_zvs
kT_vals = temperatures 


#### plot the global sigma map
fig=plt.figure(figsize=(10,8))
fig.subplots_adjust(wspace=0.1, hspace=0.1,left=0.15, right=0.95,
                        bottom=0.1, top=0.9)
gs = gridspec.GridSpec(1, 1,hspace=0)
ax=plt.subplot(gs[0])

X, Y = np.meshgrid(zv_vals, kT_vals)
Z = global_sigma_map
c=ax.contourf(X, Y, Z,  25,origin='upper',cmap='Blues')
k=fig.colorbar(c,ax=ax)
k.ax.tick_params(labelsize=20)
k.ax.set_ylabel(r'Significance ($\sigma$)',fontsize=20)
ax.set_title("Global Sigma Map (Look-Elsewhere)",fontsize=25)
ax.set_xlabel('Velocity (c)',fontsize=25)
ax.set_ylabel("kT (keV)",fontsize=25)
ax.set_yscale("log")
ax.invert_xaxis()
ax.tick_params('both', length=8, width=1, which='major',direction='in',labelsize=20)
ax.tick_params('both', length=3, width=1, which='minor',direction='in',labelsize=20)

max_glob_idx = np.unravel_index(np.argmax(global_sigma_map), global_sigma_map.shape)
kT_max_glob = kT_vals[max_glob_idx[0]]
zv_max_glob = zv_vals[max_glob_idx[1]]
print(kT_max_glob,zv_max_glob)
ax.errorbar(zv_max_glob,kT_max_glob,fmt='+',color='red',markersize=15,markeredgewidth=4)


plt.tight_layout()
plt.show()
