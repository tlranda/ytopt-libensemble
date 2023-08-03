#!/bin/bash -x
#PBS -l walltime=06:00:00
#PBS -l select=65:system=polaris
#PBS -l filesystems=home
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 64 );
mpi_ranks=( 4 8 16 32 64 );
n_nodes=(   1 2  4  8 16 );
#mpi_ranks=( 4 8 16 32 64 256 492 );
#  nodes =   1 2  4  8 16  64 123
workers=( 4 );
# MAX nodes = 4 * 16 + 1 = 65 (13% cluster capacity)
# POLARIS has MAX 496 on PROD queue
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
while [[ ${#str_nodes} -lt 2 ]]; do # Must have this number equal number of places to pad (ie: 2 == ##, 4 == ####)
    str_nodes="0${str_nodes}";
done;
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --gpu-enabled --ensemble-workers ${n_workers} --max-evals 30 --configure-environment craympi --machine-identifier polaris-gpu --system polaris --ens-dir-path Polaris_TL_apps_${n_ranks}r_${app_scale}a --ens-template run_gctla.py --ens-script qsub_tl.batch --gc-input logs/PolarisSourceTasks/*${str_nodes}n_*a/manager_results.csv --gc-ignore logs/PolarisSourceTasks/Polaris_${str_nodes}n_${str_app}a/manager_results.csv --gc-initial-quantile 0.8 --launch-job --display-results";
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

