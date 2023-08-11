#!/bin/bash -x
#COBALT -t 06:00:00
#COBALT -n 127
#COBALT --attrs filesystems=home
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

export IBV_FORK_SAFE=1;

echo "${HOSTNAME}";
date;
echo;

python3 variance_replay.py;
echo;
date;
echo;

