#!/bin/bash -x

#PBS -l walltime=03:00:00
#PBS -l select=10:system=polaris
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_gpu_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

for n_workers in `seq 1 10`; do
    echo "Calling on ${n_workers} workers";
    call="python libEwrapper.py --worker-nodes 8 --worker-timeout 300 --application-scale 256 --ensemble-workers ${n_workers} --max-evals $(( ${n_workers} * 20 )) --machine-identifier polaris-gpu --ensemble-dir-path libE_Scaling_${n_workers} --launch-job";
    echo "${call}";
    #eval "${call}";
done;

