#!/bin/bash -x

#COBALT -t 01:00:00
#COBALT -n 32
#COBALT --attrs filesystems=home,grand,eagle
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

for n_workers in `seq 1 10`; do
    echo "Calling on ${n_workers} workers";
    call="python libEwrapper.py --mpi-ranks 8 --worker-timeout 300 --application-scale 64 --cpu-override 256 --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier theta-knl --ensemble-dir-path ThetaScaling_${n_workers} --ensemble-path-randomization --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
done;

