#!/usr/bin/env python
#run source $SPEX90/spexdist.sh first!
import numpy as np
import os
import subprocess
import multiprocessing as mp
import shutil


### Set up the parameter grids you want to scan, e.g. logxi from -3 to 3 with 21 points i.e. step=0.3; and velocity from -60000km/s to 0km/s with 121 points, i.e. step=500km/s
xi_vals=np.linspace(-3,3,21)
zv_vals=np.linspace(-60000,0,121)

##parallelization function
def jobsub_call(fname):
    subprocess.call(["bash %s" % (fname)], shell=True)
    pass

### Set up the number of parallel tasks
Ncpus=int(10)
### Set up the number of cores you will use for each task
os.environ["OMP_NUM_THREADS"] = "1"

pool=mp.Pool(Ncpus)

i=0
### create the work directory 
grid_dir='xabs_202407'
if os.path.exists(grid_dir):
    shutil.rmtree(grid_dir)
os.makedirs(grid_dir)

### give an initial value for the line width
line_width="200" #km/s
### give the ionization balance file for XABS
ionbal="/home/yxu/1ES1927/analysis/ionbal/xabs_calculation_202407/xabs_inputfile_corr1" 
filenames=[]
for zv in zv_vals:
    ### set up a suitable initial value for the column density, otherwise SPEX will drop it to zero
    for xi in xi_vals:
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
        ### remove the tiny float numbers of each grid, though not necessary
        zv=round(zv)
        xi=round(xi,4)
        modstr="zv%s_xi%s" % (str(zv),str(xi))
        fname=grid_dir+"/xabsgrid_%s.sh" % modstr
        spec_fname=grid_dir+"/xabsgrid_%s" % modstr
        if not os.path.exists(spec_fname+'.out'):
            i+=1
            print("submitting job",i)
            print(spec_fname+'.out')
            filenames.append(fname)
            outfile=open(fname,"w")

            preamble=("#!/usr/bin/env bash\n\nspex<<EOF\n\n")

            initials="\n".join(
                                ["log exe WA_comt+bb_202407_slow", \
                                "com xabs",\ 
                                "com rel 4:6 8,9,3,1,7,2",\
                                "par 1 9 co av "+ionbal,\
                                "par 1 9 xi s f",\
                                "par 1 9 v s t",\
                                "par 1 9 v r 0 1500",\
                                "par 1 9 zv r -2e5 2e5",\
                                "par 1 9 zv s f"]
                                )+"\n\n"

            fixed_pars="\n".join(
                                ["par 1 9 v v "+line_width]
                                )+"\n"

            variable_pars="\n".join(["par 1 9 xi v %s" % str(xi),\
                                    "par 1 9 zv v %s" % str(zv),\
                                    "par 1 9 nh v %s" % str(nh),\
                                    "fit iter 5",\
                                    "fit iter 3",\
                                    "log out %s over" % spec_fname,\
                                    "par sho fre",\
                                    "log close out"])+"\n"

            post="quit\nEOF"

            outfile.write(preamble+initials+fixed_pars+variable_pars+post)
            outfile.flush()
            outfile.close()


pool.map(jobsub_call,filenames)

    # Loop through all files in the directory
for filename in os.listdir(grid_dir):
        # Check if the file ends with .sh
    if filename.endswith(".sh"):
            # Construct absolute file path
        file_path = os.path.join(grid_dir, filename)
            # Remove the file
        os.remove(file_path)
        print(f'Removed: {file_path}')
