#!/bin/bash

# In order for this to NOT just clean the head node, you need to launch it via mpiexec/aprun
# As arguments to this script, supply a grep expression for the process name in nvidia-smi output
# This is essentially converted to: grep -e $1 -e $2 ...

# Something along the lines of:

# mpiexec -n NODES --ppn 1 ./gpu_cleanup.sh speed3d_r2c
# aprun -n NODES -N 1 ./gpu_cleanup.sh speed3d_r2c

function join_by {
    local d=${1-} f=${2-};
    shift 1;
    printf "%s" "${@/#/$d}"
    echo $res
}

x=$(join_by " -e " ${@})
#echo "${x}"

rank=${PMI_RANK};
pids=$( nvidia-smi | grep ${x} | awk '{ print $5 }' ) #| xargs -n1 kill -9
for pid in ${pids[@]}; do
    echo "Rank ${rank} send kill to ${pid}";
    echo "${pid}" | xargs -n1 kill -9;
done;

