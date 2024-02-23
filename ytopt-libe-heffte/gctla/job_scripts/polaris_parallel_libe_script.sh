#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=65:system=polaris
#PBS -l filesystems=home
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_knight_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/gctla;
#python3 parallel_libe_driver.py --description weak_scaling_job.txt --max-nodes 64 --n-records 200 --sleep 1 --max-workers 4 --demo > parallel_driver.output;
python3 parallel_libe_driver.py --description weak_scaling_job.txt --max-nodes 64 --n-records 200 --sleep 60 --max-workers 4 > parallel_driver.output;

