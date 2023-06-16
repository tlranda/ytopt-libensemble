#!/bin/bash -x

#PBS -l walltime=03:00:00
#PBS -l select=21:system=polaris
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_gpu_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

for n_workers in `seq 1 2`; do
    echo "Calling on ${n_workers} workers";
    call="python libEwrapper.py --mpi-ranks 8 --worker-timeout 300 --application-scale 64 --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier polaris-gpu --ensemble-dir-path ScalingWED_${n_workers} --ensemble-path-randomization --launch-job --display-results";
    echo "${call}";
    eval "${call}";
done;

