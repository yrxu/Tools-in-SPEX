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
    worker_id, grid_points, stat_com, para_com, save_dir = arguments
    session=Session()
    session.command("log exe "+stat_com)
    session.command("com file")
    session.command("com reds")
    session.command("com rel 12 13,3,1,7,2")
    session.command("par 1 13 fl v 1")
    session.command("par 1 13 z s f")
    session.command("par 1 13 z r -1 1")
    session.command("par 1 12 no s t")
    for idx, xi, zv in grid_points:
        zv=round(zv)
        xi=round(xi,2)
        modstr="zv%s_xi%s" % (str(zv), str(xi))
        spec_fname=save_dir+"/piongrid_%s" % modstr
        if xi<0:
            no=1e5 ##check before running
        elif xi>=0 and xi<1:
            no=5e5
        elif xi>=1 and xi<2:
            no=1e6
        elif xi>=2 and xi<3:
            no=5e6
        elif xi>=3 and xi<4:
            no=1e7
        elif xi>=4 and xi<5:
            no=5e7
        elif xi>=5 and xi<6:
            no=1e8
        if not os.path.exists(spec_fname+'.out'):
            print("Start to fit the grid of logxi "+str(xi)+ " and velocity "+str(zv)+"km/s.")
            session.command("log exe "+para_com)
            session.command("par 1 12 fi av "+save_dir+'/piongrid_lw100_xi'+str(xi)+'.out')
            session.command("par 1 12 no v "+str(no))
            session.command("par 1 13 z v "+str(zv/3e5))
            session.command("par sho ")
            session.command("par sho fre")
            session.command("fit iter 5")
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

# create the work directory
ID='202410'
grid_dir='pion_'+ID
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)

# input files
startup_com="/PATH-TO-YOUR-STARTUP-FILE/4WA_comt+bb_"+ID ### startup fitting file
model_para="/PATH-TO-YOUR-BEST-FIT-PARAMETERS/model_para_OM_4WA_comt+bb_"+ID ### best-fit parameters


# create the model grids to scan
xi_vals=np.linspace(0,6,31)
zv_vals=np.linspace(-30000,30000,121)

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
    worker_params.append((worker_id, grid_per_worker[worker_id], startup_com, model_para,  grid_dir))

### start to run fit in parallel
pool.map(run_spex_fit, worker_params)

print("The scan is finished.")

final_file=grid_dir+"/pion_2Df_"+ID+".dat"
pattern = r"C-statistic\s*:\s*([0-9.]+)"
with open(final_file,'w') as summary:
    summary.write("# C-ST Log_xi v (km/s) VT (km/s)\n")
    summary.write("#\n")
    for idx, xi, zv in model_grids:
        zv=round(zv)
        xi=round(xi,2)
        modstr="zv%s_xi%s" % (str(zv),str(xi))
        spec_fname=grid_dir+"/piongrid_%s" % modstr
        filepath=spec_fname+'.out'
        with open(filepath, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    c_stat_value = match.group(1)
                    summary.write(f"{c_stat_value}\t{xi}\t{zv}\t100\n")
                    break
print("The results are saved in "+final_file)
