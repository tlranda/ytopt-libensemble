#!/bin/bash
#PBS -l walltime=23:00:00
#PBS -l select=256:system=polaris
#PBS -l filesystems=home
#PBS -A EE-ECP
#PBS -q prod

# N_Nodes | Queue_Name | Max_Walltime
#  10-24  |    small   |   3 hours
#  25-99  |   medium   |   6 hours
# 100-496 |    large   |  24 hours

source /home/trandall/polaris_knight_env.sh;
module list;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/gctla;
#python3 parallel_libe_driver.py --description gctla_weak_scaling_job.txt --max-nodes 256 --n-records 100 --sleep 1 --max-workers 4 --demo;
python3 parallel_libe_driver.py --description gctla_weak_scaling_job.txt --max-nodes 256 --n-records 100 --sleep 60 --max-workers 4;

