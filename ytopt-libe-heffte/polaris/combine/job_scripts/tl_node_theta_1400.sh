#!/bin/bash -x
#COBALT -t 06:00:00
#COBALT -n 513
#COBALT --attrs filesystems=home
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 1400 );
mpi_ranks=( 64 128 256 512 1024 4096 8192 );
n_nodes=(    1   2   4   8   16   64  128 );
# nodes =    1   2   4   8   16   64  128
workers=( 4 );
# MAX nodes = 128 * 4 + 1 = 513 (12% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;

for app_scale in ${app_scales[@]}; do
str_app=$app_scale
while [[ ${#str_app} -lt 4 ]]; do # Must have this number equal number of places to pad (ie: 2 == ##, 4 == ####)
    str_app="0${str_app}";
done;
for rank_index in ${!mpi_ranks[@]}; do
n_ranks=${mpi_ranks[$rank_index]};
n_nodes=${n_nodes[$rank_index]};
str_nodes=$n_nodes;
while [[ ${#str_nodes} -lt 3 ]]; do # Must have this number equal number of places to pad (ie: 2 == ##, 4 == ####)
    str_nodes="0${str_nodes}";
done;
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --cpu-override 256 --cpu-ranks-per-node 64 --ensemble-workers ${n_workers} --max-evals 30 --configure-environment craympi --machine-identifier theta-knl --system theta --ens-dir-path Theta_TL_nodes_${n_ranks}r_${app_scale}a --ens-template run_gctla.py --ens-script qsub_tl.batch --gc-input logs/ThetaSourceTasks/*n_${str_app}a/manager_results.csv --gc-ignore logs/ThetaSourceTasks/Theta_${str_nodes}n_${str_app}a/manager_results.csv --gc-initial-quantile 0.8 --launch-job --display-results";
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

