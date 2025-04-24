#!/usr/bin/python
import corner
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from nautilus import Prior, Sampler
from pyspex import Session
#from xspec import *

os.environ["OMP_NUM_THREADS"] = "1"

### SPEX set-up
s = Session()
s.command('log exe test_MCMC_202403')
s.calc()
model1 = "test_nautilus"


### Set-up priors and likelihood (i.e. -1/2 * fit statistic)
prior = Prior()
prior.add_parameter('Gamma', dist=(2, 3))
prior.add_parameter('norm', dist=(8, 8.5))

def likelihood(param_dict):
    para1=param_dict['Gamma']
    para2=10**float(param_dict['norm'])
    s.command(f'par 1 3 ga val {para1}')
    s.command(f'par 1 3 norm val {para2}')
    s.command('calc')
    cstat = float(s.fit_cstat()[0])
    return -0.5*cstat

### Run the sampler (not sure what the optimal set-up is yet, most inputs are default or lower for testing)
sampler = Sampler(prior, likelihood, n_live=1000, n_networks=4, vectorized=False, pool=4, filepath='test_nautilus.h5',resume=True)

start = time.perf_counter()
sampler.run(verbose=True, f_live=0.1, n_eff=10000,discard_exploration=True)
end = time.perf_counter()

### Get posteriors, plot corner plots (THIS DOES NOT SAVE ANYTHING...yet)
points, log_w, log_l = sampler.posterior()
corner.corner(
    points,
    weights=np.exp(log_w),
    bins=20,
    labels=prior.keys,
    range=np.repeat(0.999, len(prior.keys)),
    color='purple',
    plot_datapoints=False,
    levels = (0.393, 0.683, 0.955, 0.997,),
    plot_density = True,
    fill_contours = True,
    quantiles = [0.159, 0.500, 0.841],
    show_titles = True,
    title_fmt = '.4f',
    title_kwargs = {"fontsize":12},
    verbose = True
    )

### Print logZ and time taken to the terminal
print('log Z: {:.2f}'.format(sampler.log_z))
print(f"{end-start} seconds")


plt.show()
                                     
