#!/bin/bash -x
#COBALT -t 06:00:00
#COBALT -n 257
#COBALT --attrs filesystems=home
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;
python3 parallel_libe_driver.py --max-nodes 256 --searched logs/ThetaSourceTasks/Theta_*n_*a --n-records 400;

