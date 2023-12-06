#!/bin/bash 
#COBALT -t 06:00:00
#COBALT -n 513
#COBALT --attrs filesystems=home
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

# App and MPI should scale together as weak-scalars
app_scales=( 64 128 256  512 1024 1400 1400 );
mpi_ranks=( 128 256 512 1024 2048 4096 8192 );
n_nodes=(     2   4   8   16   32   64  128 );
# nodes =     2   4   8   16   32   64  128
n_workers=4;
# MAX nodes = 128 * 4 + 1 = 513 (12% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;

# Form the dataset to use as inputs
dataset_basis="logs/ThetaSourceTasks/Theta_"
weak_dataset=( );
for rank_index in ${!app_scales[@]}; do
    str_app=${app_scales[$rank_index]};
    while [[ ${#str_app} -lt 4 ]]; do
        str_app="0${str_app}";
    done;
    str_nodes=${n_nodes[$rank_index]};
    while [[ ${#str_nodes} -lt 3 ]]; do
        str_nodes="0${str_nodes}";
    done;
    weak_dataset+=( "${dataset_basis}${str_nodes}n_${str_app}a/manager_results.csv" );
done;

for rank_idx in ${!weak_dataset[@]}; do
    left_out=${weak_dataset[$rank_idx]};
    n_ranks=${mpi_ranks[$rank_idx]};
    n_nodes=${n_nodes[$rank_idx]};
    app_scale=${app_scales[$rank_idx]};
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks (${n_nodes} nodes) per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --cpu-override 256 --cpu-ranks-per-node 64 --ensemble-workers ${n_workers} --max-evals 30 --configure-environment craympi --machine-identifier theta-knl --system theta --ens-dir-path Theta_Weak_TL_${n_nodes}n_${app_scale}a --ens-template run_gctla.py --ens-script qsub_tl.batch --gc-input ${weak_dataset[@]} --gc-ignore ${left_out} --gc-initial-quantile 1.0 --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
echo "Requested ${calls} calls";
date;

