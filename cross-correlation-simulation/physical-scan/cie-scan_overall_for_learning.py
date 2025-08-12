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


### 3) cross-correlation
model_file = work_dir+"/simulated_model_spectra_kT"+str(len(temperatures))+"_zv"+str(len(sys_zvs))+".h5"                   # simulated HDF5 file
real_residual_file = spec_fname_real+".qdp"   # real residual spectrum
sim_residual_file = work_dir+"/merge_res_"+str(number)+".txt"     # simulated residual spectrum
output_dir = work_dir+'/'                   # output dir

### read real and simulated data
real_energy, real_flux = extract_first_instrument(real_residual_file)
sim_data = np.loadtxt(sim_residual_file)
sim_energy = sim_data[:, 0]
sim_flux_all = sim_data[:, 1:]  # shape = (n_energy, N_sim)

### read the model file
with h5py.File(model_file, "r") as f:
    model_energy = f["energy"][:]      # (n_energy,)
    kT_vals      = f["kT"][:]          # (nkT,)
    zv_vals      = f["zv"][:]/3e5          # (nzv,)
    models       = f["flux"][:]      # shape: (nkT, nzv, n_energy)

nkT, nzv, n_energy = models.shape
n_sims = sim_flux_all.shape[1]

### check the consistency between the X-axis
if not (np.allclose(real_energy, model_energy) and np.allclose(sim_energy, model_energy)):
    raise ValueError("Energy grids are not aligned!")

### 3-1-1 real residuals vs models
cc_real = np.zeros((nkT, nzv))
for i in range(nkT):
    for j in range(nzv):
        cc_real[i, j] = cross_correlation(real_flux, models[i, j])

### 3-1-2 sim residual vs models
cc_sim = np.zeros((nkT, nzv, n_sims))
for i in range(nkT):
    for j in range(nzv):
        for s in range(n_sims):
            cc_sim[i, j, s] = cross_correlation(sim_flux_all[:, s], models[i, j])

### 3-2 normalization
cc_real_norm = np.zeros_like(cc_real)
cc_sim_norm = np.zeros_like(cc_sim)

for i in range(nkT):
    for j in range(nzv):
        norm_factor = np.sqrt(np.sum(cc_sim[i, j] ** 2) / n_sims)
        cc_real_norm[i, j] = cc_real[i, j] / norm_factor
        cc_sim_norm[i, j]  = cc_sim[i, j] / norm_factor

### 3-3 calculate the significance
p_min = 1.0 / n_sims
sigma_max = pvalue_to_sigma(p_min)  # maximal sigma
print(f"Maximal sigma: {sigma_max:.2f}")

### 3-3-1 calculate the local sigma map
local_sigma_map = np.zeros((nkT, nzv))
for i in range(nkT):
    for j in range(nzv):
        sim_dist = cc_sim_norm[i, j, :]
        real_val = cc_real_norm[i, j]
        frac = np.sum(sim_dist >= real_val) / n_sims
        sigma = pvalue_to_sigma(frac)
        sigma = min(sigma, sigma_max)
        local_sigma_map[i, j] = sigma

### 3-3-2 calculate the global sigma map (look-elsewhere)
cc_sim_all_flat = cc_sim_norm.reshape(-1, n_sims)  # (n_grids, n_sims)
global_sigma_map = np.zeros((nkT, nzv))
cc_sim_all_max = np.max(cc_sim_all_flat,axis=0)
for i in range(nkT):
    for j in range(nzv):
        real_val = cc_real_norm[i, j]
        frac = np.sum(cc_sim_all_max >= real_val) / cc_sim_all_max.size
        sigma = pvalue_to_sigma(frac)
        sigma = min(sigma, sigma_max)
        global_sigma_map[i, j] = sigma

### save the output
out_local_sigma    = work_dir+"/local_sigma_map.npy"
out_global_sigma   = work_dir+"/global_sigma_map.npy"

np.save(out_local_sigma, local_sigma_map)
np.save(out_global_sigma, global_sigma_map)
print("Local sigma map saved to "+out_local_sigma)
print("Global sigma map saved to "+out_global_sigma)


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
