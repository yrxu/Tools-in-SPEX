#!/bin/bash

### This file is modified by XSPEC_absorption_grid.sh in Tools-in-XSPEC repository to collect results generated by XABS-SCAN-Parallel.py. You can write your own python file or codes at the end of XABS-SCAN-Parallel.py.

SOURCE=1ES1927 # Provide the source name: IMPORTANT is often used as the PATH-TO-DIR

type=xabs          # among: xabs pion cie hot (and other SPEX plasma models)



### keep the same grids with XABS-SCAN-Parallel.py
xi_min=-3.0
xi_max=3.01    
xi_step=0.3
zv_min=0           
zv_max=60000      
width_list=(100 250 500 1000 2500 5000 10000) # Line widths to be adopted in km/s
vstep_list=(300 300 500 700 1000 2500 2500)   # Velocity steps to adopted in km/s
N_of_width=3       # ="${#width_list[@]}" for all line widths (how many to run)
min_item=2         # =0 to start from 1st of the line widths (from which to start)

ID=(202407)
for u in 0
do
DIR_home=${PWD}/xabs_${ID[$u]}  # Launch routine from current directory or change here
DIR_outgrid=${DIR_home}/

max_item=$(( ${N_of_width}-1 ))

for item in $(seq ${min_item} 1 ${max_item}) # 1) LOOP OVER LINE WIDTH
do

width=${width_list[${item}]}
zv_step=${vstep_list[${item}]}

results_file_1=${DIR_home}/${type}_${width}_kms_${part}_2Df.dat    # OUTPUT 1: 2col file
echo "# C-ST  NH_${type}   Log_xi  v(km/s) VT (km/s)" >  ${results_file_1}
echo "# "                                             >> ${results_file_1}
results_file_2=${DIR_home}/part_${part}/${type}_${width}_kms_${part}_CS_3Df.dat # OUTPUT 2: CS matrix
results_file_3=${DIR_home}/part_${part}/${type}_${width}_kms_${part}_NH_3Df.dat # OUTPUT 3: NH matrix

for xi_start in $(seq ${xi_min} ${xi_step} ${xi_max}) # 2) LOOP OVER LOG_XI
do

################# READ THE SPEX RESULTS IN ONE FILE PER LINE WIDTH  ###############################

for j in $(seq ${zv_min} ${zv_step} ${zv_max})  # 3) LOOP OVER ZV_LOS
do

k=-${j}

input_file=${DIR_outgrid}/${type}grid_zv${k}_xi${xi_start}.out # individual fit results to read

if [ ! -f "${input_file}" ]
then
test_command=0 # if file for a certain point does not exist, ignore it or comment next line to track
else

# Removing useless lines containing e.g. "<=>" from ${input_file} to save space in your pc!

sed -i '/<->/d'        ${input_file} # If it doesnt work in Linux remove the option -i
sed -i '/bb/d'         ${input_file} # You might have to add/remove '' after option -i
sed -i '/Instrument/d' ${input_file}
sed -i '/sect/d'       ${input_file}
sed -i '/Flux/d'       ${input_file}
sed -i '/phot/d'       ${input_file}
sed -i '/----/d'       ${input_file}
sed -i '/^$/d'         ${input_file}

CSTAT=`sed '/C-statistic       :/!d'     ${input_file}` # search line containing CSTAT parameter
NORM=`sed '/ 1   9 xabs nh/!d'          ${input_file}` # IMPORTANT: Needs exact line (5 xabs)


echo ${CSTAT:24:12} ${NORM:41:13} ${xi_start} ${k} ${width} >> ${results_file_1} # update output1
echo -n ${CSTAT:24:12} " " >> ${results_file_2} # update output file 2
echo -n ${NORM:41:13}  " " >> ${results_file_3} # update output file 3
fi

done

echo " " >> ${results_file_2} # output2: put a space between each point
echo " " >> ${results_file_3} # output3: put a space between each point
done
done
cd ${DIR_home}
done
###################################### END OF THE ROUTINE ############################################
