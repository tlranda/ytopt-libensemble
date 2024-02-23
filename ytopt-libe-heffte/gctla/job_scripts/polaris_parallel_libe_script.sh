#! /bin/bash

source /home/trandall/polaris_knight_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/gctla;
python3 parallel_libe_driver.py --description weak_scaling_job_start.txt --max-nodes 2 --n-records 4 --sleep 1 --max-workers 1;
