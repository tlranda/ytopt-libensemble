#!/bin/bash -x
#COBALT -t 06:00:00
#COBALT -n 257
#COBALT --attrs filesystems=home,grand,eagle
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 1400 );
mpi_ranks=( 64 128 256 512 1024 4096 8192 16384 );
# nodes =    1   2   4   8   16   64  128   256
workers=( 4 );
# MAX nodes = (64 * 4 + 1) == (256 + 1) = 257 (<6% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;
for app_scale in ${app_scales[@]}; do
for n_ranks in ${mpi_ranks[@]}; do
for n_workers in ${workers[@]}; do
    if [[ ${n_ranks} -ge 8192 ]]; then
        echo "Downscale to 1 worker for big job";
        n_workers=1;
    fi;
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --cpu-override 256 --cpu-ranks-per-node 64 --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier theta-knl --system theta --ens-dir-path Theta_${n_ranks}r_${app_scale}a --launch-job --display-results";
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

