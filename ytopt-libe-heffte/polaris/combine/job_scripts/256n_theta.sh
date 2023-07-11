#!/bin/bash -x
#COBALT -t 06:00:00
#COBALT -n 257
#COBALT --attrs filesystems=home,grand,eagle
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 64 128 256 512 1024 );
mpi_ranks=( 16384 );
#mpi_ranks=( 64 128 256 512 1024 2048 4096 8192 16384 );
# nodes =    1   2   4   8   16   32   64  128    256
workers=( 1 );
# MAX nodes = 257 (<6% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;
for app_scale in ${app_scales[@]}; do
for n_ranks in ${mpi_ranks[@]}; do
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --cpu-override 256 --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier theta-knl --ensemble-dir-path Theta_${n_ranks}r_${app_scale}a --ensemble-path-randomization --libensemble-export libE_Run_.py --libensemble-randomization --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
done;
echo;
done;
echo "Requested ${calls} calls";

