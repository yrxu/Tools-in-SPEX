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
def simulate_physical_model_save(arguments):
    worker_id, grid_points, da_com, reds, linewidth, save_dir = arguments
    session=Session()
    session.command("log exe "+da_com)
    session.command("com cie")
    session.command("com reds")
    session.command("com reds")
    session.command("com rel 1 2,3")
    session.command("par 1 2 fl v 1")
    session.command("par 1 2 z s f")
    session.command("par 1 1 it couple 1 1 t")
    session.command("par 1 1 t s f")
    session.command("par 1 1 no s f")
    session.command("par 1 3 fl v 0")
    session.command("par 1 3 z v " + str(reds))
    for idx, t, zv in grid_points:
        f_name="cie_kT"+str(round(t,5))+"keV_zv"+str(round(zv,5))
        spec_fname=save_dir+"/%s" % f_name
        if not (os.path.exists(spec_fname+'.qdp') and os.path.exists(spec_fname+'_ign.qdp')):
            z=zv/300000
            session.command("par 1 1 v v "+ str(linewidth))
            session.command("par 1 1 t v "+str(t))
            session.command("par 1 1 no v 1") ### don't change this value
            session.command("par 1 2 z v "+str(z))
            session.calc()
            plot_save_model(session=session,filename=spec_fname)
            session.command("ion ign all")
            session.calc()
            plot_save_model(session=session,filename=spec_fname+"_ign")
            session.command("ion use all") ### simulate model w and wo lines to get the continuum-subtracted model afterwards
    session.__del__()
    print("Worker"+str(worker_id)+" finished "+str(len(grid_points))+" model simulations.")

##extract residual files
def extract_first_instrument(file_path):
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

### merge model spectra to h5 file
def merge_spectra_to_hdf5(folder, zv_vals, kT_vals, output_filename="simulated_model_spectra.h5"):
    spectra = []
    coords = []
    x_template = None
    for zv in zv_vals:
        for kT in kT_vals:
            file1 = folder+"/cie_kT"+str(round(kT,5))+"keV_zv"+str(round(zv,5))+".qdp"
            file2 = folder+"/cie_kT"+str(round(kT,5))+"keV_zv"+str(round(zv,5))+"_ign.qdp"
            if not (file1 and file2):
                raise ValueError("Cannot find the files.")

            xs, ys1 = extract_first_instrument_model(file1)
            _ , ys2 = extract_first_instrument_model(file2)
            ys = ys1 - ys2 #### remove the continuum contribution

            if x_template is None:
                x_template = xs
            else:
                if not np.allclose(x_template, xs):
                    raise ValueError(f"the X-axis data in {file} do not match with others")

            spectra.append(ys)
            coords.append((kT, zv))

    spectra = np.array(spectra)
    kTs, zvs = zip(*coords)

    # get the unique value and re-order
    kTs_unique = sorted(set(kTs))
    zvs_unique = sorted(set(zvs))

    # create empty array
    y_array = np.empty(( len(kTs_unique), len(zvs_unique), len(x_template)))

    # match the spectra
    for idx, (kT, zv) in enumerate(coords):
        i = kTs_unique.index(kT)
        j = zvs_unique.index(zv)
        y_array[i, j, :] = spectra[idx]

    # create xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            flux=(["kT", "zv",  "energy"], y_array),
        ),
        coords=dict(
            kT=kTs_unique,
            zv=zvs_unique,
            energy=x_template,
        ),
    )

    # save in the format of HDF5
    ds.to_netcdf(work_dir+'/'+output_filename, engine="h5netcdf")
    print(f"saved to {output_filename}")

### simple cross-correlation
def cross_correlation(y1, y2):
    return np.sum(y1 * y2)  
    
##p-value to sigma
def pvalue_to_sigma(p):
    """Two-tailed p-value conversion."""
    return -stats.norm.ppf(p / 2)

### set up the CPU numbers
Ncpus=int(10)
os.environ["OMP_NUM_THREADS"] = "1"
pool=mp.Pool(Ncpus,maxtasksperchild=1)
print("Running in parallel on",Ncpus,"CPUs")

### create the work directory
ID='202403'
work_dir='CC-MC-cie_'+ID
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
number=1000 #number of simulations
redshift=0.019422 # the redshift of the target
lw= 100 # linewidth of the CIE model
index_instrument="1" # simulated instrument
index_region="1" # simulated model region
startup_com="2hot+WA_comt+bb_"+ID ### startup fitting file
data_com="readdata_"+ID ### startup fitting file
model_parameter="model_para_OM_2hot+WA_comt+bb_"+ID ### best-fit parameters
# model grids
sys_zvs = np.arange(-45000,45000,step=1000)
temperatures = np.logspace(np.log10(0.2),np.log10(4),90)


### 2) create the model spectrum
model_grids=[]
idx=0

for zv in sys_zvs:
    for t in temperatures:
        model_grids.append((idx, t, zv))
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
    worker_params.append((worker_id, grid_per_worker[worker_id], data_com, redshift, lw, model_grid_dir))
pool.map(simulate_physical_model_save, worker_params)
pool.close()

###merge into the HDF5 file
merge_spectra_to_hdf5(model_grid_dir, sys_zvs, temperatures, output_filename="simulated_model_spectra_kT"+str(len(temperatures))+"_zv"+str(len(sys_zvs))+".h5")
