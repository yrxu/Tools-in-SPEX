#!/usr/bin/env python
#run source $SPEX90/spexdist.sh first!
import numpy as np
import os
import subprocess
import multiprocessing as mp
import shutil
from pyspex import Session



## function to run the SPEX in parallelization
def run_spex_fit(arguments):
    worker_id, grid_points, stat_com, para_com, ion_file, save_dir = arguments
    session=Session()
    session.command("log exe "+stat_com)
    session.command("com xabs")
    session.command("com rel 4:6 8,9,3,1,7,2")
    session.command("par 1 9 co av "+ion_file)
    session.command("par 1 9 xi s f")
    session.command("par 1 9 zv s f")
    session.command("par 1 9 v s t")
    session.command("par 1 9 v r 0 1500")
    session.command("par 1 9 zv r -2e5 2e5")
    for idx, xi, zv in grid_points:
        zv=round(zv)
        xi=round(xi,4)
        modstr="zv%s_xi%s" % (str(zv),str(xi))
        spec_fname=save_dir+"/xabsgrid_%s" % modstr
        if xi<-1:
            nh=5e-5
        elif xi>=-1 and xi<1:
            nh=1e-4
        elif xi>=1 and xi<2:
            nh=2e-4
        elif xi>=2 and xi<3:
            nh=5e-4
        elif xi>=3:
            nh=8e-4
        if not os.path.exists(spec_fname+'.out'):
            print("Start to fit the grid of logxi "+str(xi)+ " and velocity "+str(zv)+"km/s.")
            session.command("log exe "+para_com)
            session.command("par 1 9 v v 200")
            session.command("par 1 9 xi v "+str(xi))
            session.command("par 1 9 nh v "+str(nh))
            session.command("par 1 9 zv v "+str(zv))
            session.command("fit iter 5")
            session.command("fit iter 8")
            session.command("log out %s over" % spec_fname)
            session.command("par sho fre")
            session.command("log close out")
    session.__del__()
    print("Worker"+str(worker_id)+" finished "+str(len(grid_points))+" grid fits.")

### set up initials
# set up the number of cores
Ncpus=int(10)
os.environ["OMP_NUM_THREADS"] = "1"
pool=mp.Pool(Ncpus,maxtasksperchild=1)
print("Running in parallel on",Ncpus,"CPUs")

# input files
startup_com="/home/yxu/1ES1927/analysis/SPEX/WA_comt+bb_202403_slow" ### startup fitting file
model_para="/home/yxu/1ES1927/analysis/SPEX/model_para_OM_WA_comt+bb_202403_slow" ### best-fit parameters
ionization_file="/home/yxu/1ES1927/analysis/ionbal/xabs_calculation_202403/xabs_inputfile_corr1" ### xabs ionization file

# create the work directory
grid_dir='xabs_202403'
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)

# create the model grids to scan
xi_vals=np.linspace(-3,3,21)
zv_vals=np.linspace(-60000,0,121)


### prepare for the parallelization
model_grids=[]
idx=0
for xi in xi_vals:
    for zv in zv_vals:
        model_grids.append((idx,xi,zv))
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
    worker_params.append((worker_id, grid_per_worker[worker_id], startup_com, model_para, ionization_file, grid_dir))

### start to run fit in parallel
pool.map(run_spex_fit, worker_params)

print("The scan is finished.")
