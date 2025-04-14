#!/usr/bin/env python
#run source $SPEX90/spexdist.sh first!
import numpy as np
import pandas as pd
import os
import time
import subprocess
import multiprocessing as mp
import shutil
import glob
import itertools
from pyspex import Session
import scipy.stats as stats

##plot and save the residual spectrum
def plot_save_residual(session,filename):
    session.command("pl dev null")
    session.command("pl ty chi")
    session.command("pl ux a")
    session.command("pl uy fa")
    session.command("pl x lin")
    session.command("pl y lin")
    session.command("pl")
    session.command("plot adum %s over" % filename)
    session.command("pl close 1")
    print("save residual spectrum to "+ filename)


##plot and save the model spectrum
def plot_save_model(session,filename):
    session.command("pl dev null")
    session.command("pl ty da")
    session.command("pl ux a")
    session.command("pl uy a")
    session.command("pl x lin")
    session.command("pl y lin")
    session.command("pl fi di f")
    session.command("pl")
    session.command("plot adum %s over" % filename)
    session.command("pl close 1")
    print("save model spectrum to "+ filename)


###simulate residual spectra and save plot
def simulate_residual_save(arguments):
    worker_id,N_sim,start_idx,st_com,expo,save_dir,idx_inst,idx_reg=arguments
    session=Session()
    session.command("log exe "+st_com)
    session.calc()
    for i in range(N_sim):
        sim_index=start_idx+i
        f_name="simulation_%s" % str(sim_index)
        spec_fname=save_dir+"/%s" % f_name
        if not os.path.exists(spec_fname+'.qdp'):
            session.simulate(expo,inst=idx_inst,reg=idx_reg,noise=True,seed=None)
            plot_save_residual(session=session,filename=spec_fname)
    session.__del__()
    print("Worker"+str(worker_id)+" finished "+str(N_sim)+" simulations.")

###simulate model spectra and save plot
def simulate_line_model_save(arguments):
    worker_id, grid_points, da_com, save_dir = arguments
    session=Session()
    session.command("log exe "+da_com)
    session.command("com gaus")
    session.command("par 1 1 ty v 1")
    session.command("par 1 1 e s f")
    session.command("par 1 1 fw s f")
    session.command("par 1 1 w s t")
    session.command("par 1 1 aw s t")
    session.command("par 1 1 no v 1e3")
    for idx, lw, wl in grid_points:
        f_name="line_lw"+str(lw)+"kms_wl"+str(round(wl,5))
        spec_fname=save_dir+"/%s" % f_name
        if not os.path.exists(spec_fname+'.qdp'):
            lw_a=lw/300000*wl
            session.command("par 1 1 w v "+str(wl))
            session.command("par 1 1 aw v "+str(lw_a*2.355))
            session.calc()
            plot_save_model(session=session,filename=spec_fname)
    session.__del__()
    print("Worker"+str(worker_id)+" finished "+str(len(grid_points))+" model simulations.")

##extract residual files
def extract_first_instrument(file_path):
    """
    Reads a file with the expected format.
    Skips the header line, then collects data lines until the first "NO" line.
    Each valid data line is assumed to have at least 4 columns.
    Returns:
        xs: list of x values (first column)
        ys: list of y values (fourth column)
    """
    xs, ys = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Skip header (first line)
    for line in lines[1:]:
        # Stop if line starts with "NO"
        if line.strip().startswith("NO"):
            break
        parts = line.strip().split()
        # Ensure the line has enough columns (x is 1st, y is 4th)
        if len(parts) >= 4:
            xs.append(float(parts[0]))
            ys.append(float(parts[3]))
    return np.array(xs), np.array(ys)

##extract model files
def extract_first_instrument_model(file_path):
    """
    Reads a file with the expected format.
    Skips the header line, then collects data lines until the first "NO" line.
    Each valid data line is assumed to have at least 4 columns.
    Returns:
        xs: list of x values (first column)
        ys: list of y values (fourth column)
    """
    xs, ys = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Skip header (first line)
    for line in lines[1:]:
        # Stop if line starts with "NO"
        if line.strip().startswith("NO"):
            break
        parts = line.strip().split()
        # Ensure the line has enough columns (x is 1st, y is 4th)
        if len(parts) >= 4:
            xs.append(float(parts[0]))
            ys.append(float(parts[6]))
    return np.array(xs), np.array(ys)

##cross-correlation function for parallelization
def func(params):
        a=params[0]
        b=params[1]
        return np.correlate(a,b)

##p-value to sigma
def pvalue_to_sigma(p):
    """Two-tailed p-value conversion."""
    return -stats.norm.ppf(p / 2)

### set up the CPU numbers
Ncpus=int(5)
os.environ["OMP_NUM_THREADS"] = "1"
pool=mp.Pool(Ncpus,maxtasksperchild=1)
print("Running in parallel on",Ncpus,"CPUs")
### create the work directory
work_dir='CC-MC_202403_python_test2'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
grid_dir=work_dir+'/simulated_spectra'
if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)
model_grid_dir=work_dir+'/model_spectra'
if not os.path.exists(model_grid_dir):
    os.makedirs(model_grid_dir)
### initial setup
exposure=85655 #seconds
number=100 #number of simulations
index_instrument="1" # simulated instrument
index_region="1" # simulated model region
startup_com="comt+bb_202403" ### startup fitting file
data_com="readdata_202403" ### startup fitting file
model_parameter="model_para_OM_comt+bb_202403" ### best-fit parameters
# model grids
linewidths = np.linspace(0, 500, 2)
wavelengths = np.arange(7,37.01,step=0.1)





### 1) Get real and simulated residuals
#save the real residual file
modstr_real="real_residual"
spec_fname_real=work_dir+"/%s" % modstr_real


s=Session()
s.command("log exe "+startup_com)
s.calc()
plot_save_residual(session=s,filename=spec_fname_real)
s.__del__()

### simulate the residual spectra
#determine number of simulations per CPU
quotient, remainder=divmod(number,Ncpus)
#build argument list for each worker along with a start index for unique filenames
args=[]
start_index=0
for worker_id in range(Ncpus):
    num_simulations = quotient + 1 if worker_id < remainder else quotient
    args.append((worker_id,num_simulations,start_index,startup_com,exposure,grid_dir,index_instrument,index_region))
    start_index += num_simulations
pool.map(simulate_residual_save,args)
pool.close()

### merge residual spectra into one file
# Get list of all files (adjust the glob pattern as needed)
file_list = glob.glob(grid_dir+"/*.qdp")
# Lists to collect the x data and the y columns from each file
x_common = None
y_columns = []
for idx, filename in enumerate(file_list):
    xs, ys = extract_first_instrument(filename)
    # For the first file, store x values
    if x_common is None:
        x_common = xs
    else:
        # Optionally, check that x values match the common x
        if not np.allclose(x_common, xs):
            raise ValueError(f"x values in {filename} do not match the common x values.")
    y_columns.append(ys)
    print(f"Processed {filename}: extracted {len(ys)} points.")
# Merge x and all y columns into one 2D array:
# First column is x, following columns are y from each file.
merged_data = np.column_stack([x_common] + y_columns)
# Save merged data to a new file. Adjust format if necessary.
np.savetxt(work_dir+"/merge_res_"+str(number)+".txt", merged_data, fmt="% .9f", comments='')
print("Merged data saved to "+work_dir+"/merge_res_"+str(number)+".txt")

