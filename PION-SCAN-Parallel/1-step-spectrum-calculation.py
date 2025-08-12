#!/usr/bin/env python
#run source $SPEX90/spexdist.sh first!
import numpy as np
import os
import subprocess
import multiprocessing as mp
import shutil
from pyspex import Session
import re

## function to run the spex in parallelization
def run_spex_fit(arguments):
    worker_id, grid_points, redshift, lw, norm, sed_file, save_dir = arguments
    session=Session()
    session.command("distance "+str(redshift)+" z")
    session.command("com file")
    session.command("com pion")
    session.command("com etau")
    session.command("com rel 1 2,3")
    session.command("par 1 1 no v " + str(norm))
    session.command("par 1 1 file av "+sed_file)
    session.command("par 1 2 xi r -3 6")
    session.command("par 1 2 om v 1")
    session.command("par 1 2 fc v 0")
    session.command("par 1 2 hd v 1e-14")
    session.command("par 1 2 v v "+str(lw))
    session.command("par 1 3 tau0 v 1e5")
    session.command("par 1 3 a v 0")
    for idx, xi in grid_points:
        #zv=round(zv)
        xi=round(xi,2)
        modstr="xi%s" % str(xi)
        spec_fname=save_dir+"/piongrid_lw"+str(lw)+"_%s" % modstr
        if not os.path.exists(spec_fname+'.qdp'):
            print("Start to fit the grid of logxi "+str(xi))
            #session.command("log exe "+para_com)
            session.command("par 1 2 xi v "+str(xi))
            session.command("cal")
            plot_save_model(session=session,filename=spec_fname)
    session.__del__()

### set up initials
# set up the number of cores
Ncpus=int(10)
os.environ["OMP_NUM_THREADS"] = "1"
pool=mp.Pool(Ncpus,maxtasksperchild=1)
print("Running in parallel on",Ncpus,"CPUs")

# create the work directory
ID='202410'
grid_dir='pion_'+ID
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)
  
# input files
sed_file="/PATH-TO-SED/PION_SED_1ES1927_202410_keV_photonserg.out"

# create the model grids to scan
norm=8.1302938E+09 #check before running
redshift=0.019422
lw=100
xi_vals=np.linspace(-3,6,91)


### prepare for the parallelization
model_grids=[]
idx=0
for xi in xi_vals:
    model_grids.append((idx,xi))
    idx+=1
N_model_grids=len(model_grids)
print("number of model grids "+ str(N_model_grids))

# Split the grid among the workers in a round-robin fashion.
grid_per_worker = [[] for _ in range(Ncpus)]
for i, grid_point in enumerate(model_grids):
    grid_per_worker[i % Ncpus].append(grid_point)
  
# Build the argument list for the workers.
worker_params = []
for worker_id in range(Ncpus):
    worker_params.append((worker_id, grid_per_worker[worker_id], redshift, lw, norm, sed_file, grid_dir))

### start to run fit in parallel
pool.map(run_spex_fit, worker_params)

print("The calculation is finished.")

for idx, xi in model_grids:
    xi=round(xi,2)
    modstr="xi%s" % str(xi)
    spec_fname=grid_dir+"/piongrid_lw"+str(lw)+"_%s" % modstr
    convert_qdp_to_out(spec_fname)

print("The conversion is done.")
