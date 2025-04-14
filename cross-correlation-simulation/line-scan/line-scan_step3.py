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


    
### 3) collect the results: -1- cross-correlate the real and simulated residuals with model spectra
###                         -2- normalize the cross-correlation with simulated dataset
###                         -3- calculate the single trial significance
###                         -4- calculate the significance considering the look-elsewhere effect

##cross-correlate real residuals
real_res=spec_fname_real+".qdp"
energy_real, y_real = extract_first_instrument(real_res)
for u in range(len(linewidths)):
    model_file=work_dir+"/merge_line_model_lw"+str(linewidths[u])+".txt"
    model_data = np.loadtxt(model_file)
    energy_model = model_data[:, 0]
    if not np.allclose(energy_real, energy_model):
        raise ValueError("Energy axes do not match between real and model spectra.")
    n_models = model_data.shape[1] - 1
    correlate=[]
    for i in range(n_models):
        y_model = model_data[:, i+1] ###avoid the energy column
        cor=np.correlate(y_real,y_model)
        correlate.append(cor)
    np.savetxt(work_dir+'/raw_correlate_real_lw'+str(linewidths[u])+'.txt',np.column_stack([wavelengths,np.array(correlate)]), fmt='%.9f')

    ##cross-correlate simulated residuals
    sim_file=work_dir+"/merge_res_"+str(number)+".txt"
    sim_data=np.loadtxt(sim_file)
    paramlist=list(itertools.product(sim_data[:,1:].T,model_data[:,1:].T))
    correlate_sim=pool.map(func,paramlist)
    correlate_stack=np.array(correlate_sim).reshape(number,n_models)
    np.savetxt(work_dir+'/raw_correlate_sim_lw'+str(linewidths[u])+'.txt',np.column_stack([wavelengths,np.array(correlate_stack).T]), fmt='%.9f')

    ##renormalized the real and each simulated cross-correlation results by simulated cross-correlations
    ##The normalization follows Kosec, Peter et al. 2021, renormalizing positive and negative values separately.
    raw_file=work_dir+'/raw_correlate_sim_lw'+str(linewidths[u])+'.txt'
    raw_data=np.loadtxt(raw_file)
    N_corr=[]
    N_corr_real=[]
    for i in range(n_models):
        corr=raw_data[i,1:]
        count_pos=np.sum([1 if c>=0 else 0 for c in corr])
        count_neg=np.sum([1 if c<0 else 0 for c in corr])
        index_pos=[True if c>=0 else False for c in corr]
        index_neg=[True if c<0 else False for c in corr]
        if count_pos==0:
            norm_pos=np.infty
        else:
            norm_pos=np.sqrt(sum([k**2 for k in corr[index_pos]])/count_pos)
        if count_neg==0:
            norm_neg=np.infty
        else:
            norm_neg=np.sqrt(sum([k**2 for k in corr[index_neg]])/count_neg)
        n_corr=[c/norm_pos if c>=0 else c/norm_neg for c in corr]
        N_corr.append(n_corr)
        n_corr_real=[c/norm_pos if c>=0 else c/norm_neg for c in correlate[i]]
        N_corr_real.append(n_corr_real)
        print('calculate the normalized significance of the '+str(i)+'th model point')
    np.savetxt(work_dir+'/norm_correlate_sim_lw'+str(linewidths[u])+'.txt',np.column_stack([wavelengths,np.array(N_corr)]), fmt='%.9f')
    np.savetxt(work_dir+'/norm_correlate_real_lw'+str(linewidths[u])+'.txt',np.column_stack([wavelengths,np.array(N_corr_real)]), fmt='%.9f')

    ## calculate the p-values and signficance
    norm_real_file=work_dir+'/norm_correlate_real_lw'+str(linewidths[u])+'.txt'
    norm_real_data=np.loadtxt(norm_real_file)
    norm_sim_file=work_dir+'/norm_correlate_sim_lw'+str(linewidths[u])+'.txt'
    norm_sim_data=np.loadtxt(norm_sim_file)
    energies=norm_real_data[:,0]
    real_corr=norm_real_data[:,1]
    sim_vals=norm_sim_data[:,1:]

    assert sim_vals.shape == (n_models, number), \
        f"Expected sim_vals shape ({n_models},{number}), got {sim_vals.shape}"
    #n_models
    #number

    ##single-trial significance
    # Sign convention: +1 if real_corr>=0 else -1
    signs = np.where( real_corr >= 0, 1.0, -1.0)
    # Count how many simulated >= real (for positive) or <= real (for negative)
    counts_pos = (sim_vals >= real_corr[:, None]).sum(axis=1)
    counts_neg = (sim_vals <= real_corr[:, None]).sum(axis=1)
    counts = np.where(real_corr >= 0, counts_pos, counts_neg)

    p_vals = counts / number
    sigmas = pvalue_to_sigma(p_vals)
    max_sigma = pvalue_to_sigma(1.0 / number)
    sigmas[np.isinf(sigmas)] = max_sigma
    single_signif = sigmas * signs
    np.savetxt(work_dir+'/single_trial_significance_lw'+str(linewidths[u])+'.txt',np.column_stack([energies,single_signif]), fmt='%.9f')
    print("save single trial significance in line with of "+str(linewidths[u]))


    ##look-elsewhere (true) significance
    max_list = sim_vals.max(axis=0)
    min_list = sim_vals.min(axis=0)
    # For each model i, count how many simulations exceed the real_corr[i] in the global sense
    counts_true_pos = (max_list[None, :] >= real_corr[:, None]).sum(axis=1)
    counts_true_neg = (min_list[None, :] <= real_corr[:, None]).sum(axis=1)
    counts_true     = np.where(real_corr >= 0, counts_true_pos, counts_true_neg)
    p_vals_true    = counts_true / number
    sigmas_true    = pvalue_to_sigma(p_vals_true)
    sigmas_true[np.isinf(sigmas_true)] = max_sigma
    true_signif    = sigmas_true * signs
    np.savetxt(work_dir+'/true_trial_significance_lw'+str(linewidths[u])+'.txt',np.column_stack([energies,true_signif]), fmt='%.9f')
    print("save true (lookelse-where considered) significance in line with of "+str(linewidths[u]))

print("Cross-correlation simulation has been finished.")
